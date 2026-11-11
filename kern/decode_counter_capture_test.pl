#!/usr/bin/env perl
use strict;
use warnings;
use FindBin qw($Bin);
use File::Temp qw(tempdir);
use JSON::PP;
use Storable qw(dclone);
use Test::More;
require "$Bin/decode_counter_capture.pl";

# All constructed CSV values below are CPU test fixtures, not GPU evidence.
sub rejects {
    my ($name, $call, $expected) = @_;
    my $ok = eval { $call->(); 1 };
    my $error = $@;
    ok(!$ok && $error =~ $expected, $name) or diag($error || 'unexpected success');
}
sub metric_csv {
    my ($names) = @_;
    my $text = "==123== Profiling result:\n";
    $text .= qq{"Device","Kernel","Invocations","Metric Name","Metric Description","Min","Max","Avg"\n};
    for my $name (@$names) {
        my @values = $name =~ /^stall_/ ? ('1.000000%', '3.000000%', '2.000000%') :
            $name eq 'achieved_occupancy' ? ('.1', '.3', '.2') : (1, 3, 2);
        $text .= join(',', map {'"'.$_.'"'} ('Tesla M60 (0)', 'die', 18, $name, 'description, with comma', @values))."\n";
    }
    return $text;
}
is_deeply(DecodeCounterCapture::csv_fields('"a,b","c""d",e,'), ['a,b','c"d','e',''], 'quoted CSV and trailing field');
rejects('reject malformed quote', sub { DecodeCounterCapture::csv_fields('"unfinished') }, qr/malformed CSV/);
my @metrics = map {@{$_->[1]}} @DecodeCounterCapture::GROUPS;
is(scalar(@metrics), 13, 'thirteen requested hardware metrics');
my $csv = metric_csv(\@metrics);
my $parsed = DecodeCounterCapture::parse_metrics($csv, \@metrics, 18);
is(scalar(keys %$parsed), 13, 'complete workload metric fixture parsed');
is($parsed->{sysmem_read_bytes}{avg}{value}, 2, 'nvprof average not divided by invocation count again');
is($parsed->{stall_sync}{avg}{unit}, 'percent', 'stall percentage retains its unit');
is($parsed->{achieved_occupancy}{avg}{unit}, 'ratio', 'occupancy is a ratio');
rejects('enumeration is not a workload measurement', sub {
    DecodeCounterCapture::parse_metrics("sysmem_read_bytes: Number of bytes\n", \@metrics, 18);
}, qr/missing workload/);
rejects('permission failure is not empty successful traffic', sub {
    DecodeCounterCapture::parse_metrics("==123== ERR_NVGPUCTRPERM\n$csv", \@metrics, 18);
}, qr/unavailable/);
rejects('reject incomplete kernel profiling warning', sub {
    DecodeCounterCapture::parse_metrics("==123== Some kernels were not profiled\n$csv", \@metrics, 18);
}, qr/unavailable/);
for my $case (
    ['missing metric', sub {$_[0] =~ s/^.*"sysmem_read_bytes".*\n//m}, qr/missing workload/],
    ['duplicate metric', sub {$_[0] .= (split /\n/, $_[0])[2]."\n"}, qr/duplicate metric/],
    ['wrong invocations', sub {$_[0] =~ s/"18"/"17"/}, qr/invocation count/],
    ['wrong kernel', sub {$_[0] =~ s/"die"/"other_kernel"/}, qr/wrong kernel/],
    ['mixed devices', sub {$_[0] =~ s/"Tesla M60 \(0\)"/"Tesla M60 (1)"/}, qr/mixed profiler devices/],
    ['nonfinite value', sub {$_[0] =~ s/"1","3","2"/"1","3","NaN"/}, qr/nonfinite/],
    ['negative value', sub {$_[0] =~ s/"1","3","2"/"-1","3","2"/}, qr/negative/],
    ['unsupported value', sub {$_[0] =~ s/"1","3","2"/"1","3","N\/A"/}, qr/unsupported/],
    ['bad average', sub {$_[0] =~ s/"1","3","2"/"1","3","4"/}, qr/min\/avg\/max/],
    ['wrong header', sub {$_[0] =~ s/"Metric Description"/"Unknown"/}, qr/header/],
    ['wrong row width', sub {$_[0] =~ s/"1","3","2"/"1","3"/}, qr/row width/],
) {
    my $x = $csv; $case->[1]->($x);
    rejects($case->[0], sub {DecodeCounterCapture::parse_metrics($x, \@metrics, 18)}, $case->[2]);
}
for my $case (
    ['NaN', 'sysmem_read_bytes', qr/nonfinite/],
    ['Inf', 'dram_read_bytes', qr/nonfinite/],
    ['1e999', 'dram_read_bytes', qr/nonfinite/],
    ['100.01%', 'stall_sync', qr/out of range/],
    ['1.01', 'achieved_occupancy', qr/out of range/],
    ['2%', 'sysmem_read_bytes', qr/wrong metric unit/],
    ['2', 'stall_sync', qr/wrong metric unit/],
) {
    rejects("invalid metric $case->[1]=$case->[0]", sub {DecodeCounterCapture::metric_number(@$case[0,1])}, $case->[2]);
}
my $m = {nv=>2000,tail=>0,iterations=>4,seed=>4242,pattern=>'changing',expected_source_revision=>'a'x40};
my $w = {%$m,source_revision=>'a'x40,host_kv=>0,budget=>512,visible=>8000,
    logical_keys=>2048,physical_gathered_slots=>2048,warmup_launches=>10,
    cache_fnv1a64=>'a'x16,selections_fnv1a64=>'b'x16,selection_changes=>3,
    verified_output_signature_changes=>3,preflight_verified_all=>JSON::PP::true};
ok(DecodeCounterCapture::validate_workload($w,$m,0), 'complete workload fixture');
for my $case (
    ['string success', 'preflight_verified_all', 'true', qr/verify outputs/],
    ['unverified output', 'preflight_verified_all', JSON::PP::false, qr/verify outputs/],
    ['wrong source', 'source_revision', 'b'x40, qr/source\/placement/],
    ['nonnumeric placement', 'host_kv', 'nonsense', qr/source\/placement/],
    ['wrong placement', 'host_kv', 1, qr/source\/placement/],
    ['wrong budget', 'budget', 4096, qr/source\/placement/],
    ['wrong physical slots', 'physical_gathered_slots', 2052, qr/workload count/],
    ['wrong logical keys', 'logical_keys', 8000, qr/workload count/],
    ['wrong warmups', 'warmup_launches', 9, qr/workload count/],
    ['wrong visible length', 'visible', 3100, qr/workload count/],
    ['unchanging selections', 'selection_changes', 0, qr/unchanged/],
    ['unchanging output', 'verified_output_signature_changes', 0, qr/unchanged/],
    ['fractional changes', 'selection_changes', 2.5, qr/change count/],
    ['missing cache hash', 'cache_fnv1a64', undef, qr/workload identity/],
) {
    my $x = dclone($w); $x->{$case->[1]} = $case->[2];
    rejects($case->[0], sub {DecodeCounterCapture::validate_workload($x,$m,0)}, $case->[3]);
}
my $host = dclone($w); $host->{host_kv}=1;
ok(eval {DecodeCounterCapture::matched_workloads($w,$host);1}, 'placement difference permitted in paired workload');
$host->{selections_fnv1a64}='c'x16;
rejects('mismatched selection identity', sub {DecodeCounterCapture::matched_workloads($w,$host)}, qr/unmatched/);
my $rows = "iteration,gpu_launch_us,selection_fnv1a64,verified_output_fnv1a64\n";
$rows .= join('',map {"$_,123.45,".('a'x16).','.('b'x16)."\n"} 0..3);
is(scalar(@{DecodeCounterCapture::workload_rows($rows,4)}),4,'complete native output rows');
my $short=$rows; $short =~ s/3,.*\n//;
rejects('truncated workload CSV',sub {DecodeCounterCapture::workload_rows($short,4)},qr/incomplete/);
my $bad=$rows; $bad =~ s/1,123/0,123/;
rejects('duplicate workload iteration',sub {DecodeCounterCapture::workload_rows($bad,4)},qr/invalid workload/);
$bad=$rows; $bad =~ s/123.45/NaN/;
rejects('nonfinite workload timing',sub {DecodeCounterCapture::workload_rows($bad,4)},qr/invalid workload/);
$bad=$rows; $bad =~ s/123.45/0/;
rejects('nonpositive workload timing',sub {DecodeCounterCapture::workload_rows($bad,4)},qr/invalid workload/);
my $tmp=tempdir(CLEANUP=>1);
my $lock=DecodeCounterCapture::acquire_lock("$tmp/lock",0,1,undef);
ok($lock,'acquire free lock');
my $notes=0;
rejects('busy lock blocks before GPU operations',sub {
    DecodeCounterCapture::acquire_lock("$tmp/lock",0,1,sub {$notes++});
},qr/^BLOCKED:/);
is($notes,1,'blocked attempt records lock status');
close $lock;
$lock=DecodeCounterCapture::acquire_lock("$tmp/lock",0,1,undef);
ok($lock,'same inode can be acquired after release');
my $inherited=DecodeCounterCapture::inherited_lock("$tmp/lock",fileno($lock));
ok($inherited,'duplicate inherited queue lease without reopening its path');
close $inherited;
rejects('closing borrowed descriptor preserves outer lease',sub {
    DecodeCounterCapture::acquire_lock("$tmp/lock",0,1,undef);
},qr/^BLOCKED:/);
open my $other,'+>',"$tmp/unrelated" or die $!;
rejects('reject unrelated inherited descriptor',sub {
    DecodeCounterCapture::inherited_lock("$tmp/lock",fileno($other));
},qr/does not identify/);
for my $fd (0,1,2,-1,'not-a-descriptor') {
    rejects("reject invalid inherited descriptor $fd",sub {
        DecodeCounterCapture::inherited_lock("$tmp/lock",$fd);
    },qr/invalid inherited/);
}
symlink "$tmp/lock","$tmp/link" or die $!;
rejects('inherited lease rejects a symlink pathname',sub {
    DecodeCounterCapture::inherited_lock("$tmp/link",fileno($lock));
},qr/does not identify/);
rejects('separate open description cannot borrow another lock',sub {
    open my $separate,'+<',"$tmp/lock" or die $!;
    DecodeCounterCapture::inherited_lock("$tmp/lock",fileno($separate));
},qr/^BLOCKED:/);
close $other;
close $lock;
my $r=DecodeCounterCapture::run_command([$^X,'-e','print "out"; print STDERR "err"; exit 7'],5);
is_deeply([@$r{qw(exit stdout stderr)}],[7,'out','err'],'subprocess exit and both streams preserved');
$r=DecodeCounterCapture::run_command([$^X,'-e','sleep 4'],1);
is($r->{exit},124,'subprocess timeout is a failure');

# Optional replay of the actual archived smoke. This does not contact a GPU.
my $capture="$Bin/../validation/issue10/counters/workloads/smoke-8000-changing";
SKIP: {
    skip 'raw GPU smoke is not installed', 1 unless -d $capture;
    my $result=JSON::PP->new->decode(DecodeCounterCapture::read_text("$capture/status.json"));
    my ($checked,$reference)=(0,undef);
    for my $g (@DecodeCounterCapture::GROUPS) {
        my %rows;
        for my $h (0,1) {
            my $stem="$capture/$g->[0]-host$h";
            my $metrics=DecodeCounterCapture::parse_metrics(DecodeCounterCapture::read_text("$stem.nvprof.csv"),$g->[1],18);
            $rows{$h}=DecodeCounterCapture::workload_rows(DecodeCounterCapture::read_text("$stem.workload.csv"),4);
            my $command=JSON::PP->new->decode(DecodeCounterCapture::read_text("$stem.command.json"));
            die "archived profiler command did not complete\n" unless $command->{exit}==0;
            my $workload=JSON::PP->new->decode($command->{stdout});
            DecodeCounterCapture::validate_workload($workload,$result->{manifest},$h);
            DecodeCounterCapture::matched_workloads($reference,$workload) if $reference;
            $reference //= $workload;
            my $accepted=JSON::PP->new->decode(DecodeCounterCapture::read_text("$stem.accepted.json"));
            my $json=JSON::PP->new->canonical;
            die "archived metrics differ from raw profiler rows\n" unless $json->encode($metrics) eq $json->encode($accepted->{metrics});
            for my $pair (['raw_profiler',"$stem.nvprof.csv"],['raw_workload',"$stem.workload.csv"]) {
                my $actual=DecodeCounterCapture::identity($pair->[1]);
                my $expected=$accepted->{$pair->[0]};
                die "raw capture identity mismatch\n" unless $actual->{sha256} eq $expected->{sha256} && $actual->{bytes}==$expected->{bytes};
            }
            $checked++;
        }
        die "archived host/VRAM output rows differ\n" unless JSON::PP->new->canonical->encode($rows{0}) eq JSON::PP->new->canonical->encode($rows{1});
    }
    is($checked,8,'current parser validates all eight archived real profiler arms');
}
done_testing;
