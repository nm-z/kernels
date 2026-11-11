#!/usr/bin/env perl
package DecodeCounterMatrix;
use strict;
use warnings;
use Cwd qw(abs_path);
use File::Basename qw(basename dirname);
use File::Path qw(make_path);
use Getopt::Long qw(GetOptionsFromArray);
use JSON::PP;
require(dirname(abs_path(__FILE__)).'/decode_counter_capture.pl');

# CPU replay of the finite component campaign. A PASS authenticates neither
# arbitrary supplied manifests nor the full-model acceptance protocol.
our $JSON = JSON::PP->new->canonical->pretty;
our @NV = (775, 2000);
our @PATTERNS = qw(clustered even random changing);
sub load { return $JSON->decode(DecodeCounterCapture::read_text($_[0])); }
sub same {
    my ($a, $b, $label) = @_;
    die "$label differs\n" unless $JSON->encode($a) eq $JSON->encode($b);
}
sub raw_identity {
    my ($record, $path) = @_;
    die "missing raw artifact identity\n" unless ref($record) eq 'HASH' && defined($record->{path}) &&
        basename($record->{path}) eq basename($path) && defined($record->{bytes}) &&
        ($record->{sha256} // '') =~ /\A[0-9a-f]{64}\z/;
    die "linked raw artifact\n" if -l $path;
    my $actual = DecodeCounterCapture::identity($path);
    die "raw artifact identity mismatch: $path\n" unless $actual->{bytes} == $record->{bytes} &&
        $actual->{sha256} eq $record->{sha256};
    return $actual;
}
sub validate_command {
    my ($command, $manifest, $group, $host, $prefix) = @_;
    die "profiler command did not exit successfully\n" unless ref($command) eq 'HASH' &&
        defined($command->{exit}) && !ref($command->{exit}) && "$command->{exit}" eq '0';
    my $argv = $command->{argv};
    die "invalid recorded command arguments\n" unless ref($argv) eq 'ARRAY' && @$argv &&
        !(grep {!defined($_) || ref($_)} @$argv);
    for my $setting ('CUDA_DEVICE_ORDER=PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES='.$manifest->{gpu_uuid}, 'DIE_CUBIN='.$manifest->{cubin}) {
        my ($key) = split /=/, $setting, 2;
        my @values = grep {index($_, "$key=") == 0} @$argv;
        die "recorded GPU environment differs\n" unless @values == 1 && $values[0] eq $setting;
    }
    my @start = grep {$argv->[$_] eq $manifest->{profiler}} 0 .. $#$argv;
    die "profiler identity missing from command\n" unless @start == 1;
    my @suffix = @$argv[$start[0] .. $#$argv];
    my @expected = ($manifest->{profiler}, '--devices', '0', '--kernels', 'die', '--csv',
        '--metrics', join(',', @$group), '--log-file', "$prefix.nvprof.csv", $manifest->{binary},
        map {"$_"} @$manifest{qw(nv tail iterations pattern)}, "$host", "$prefix.workload.csv", "$manifest->{seed}");
    same(\@suffix, \@expected, 'recorded profiler/workload arguments');
}
sub case_replay {
    my ($dir, $nv, $pattern) = @_;
    my $result = load("$dir/status.json");
    die "case is not a completed component capture\n" unless ref($result) eq 'HASH' && ($result->{status} // '') eq 'PASS' &&
        JSON::PP::is_bool($result->{full_issue_acceptance}) && !$result->{full_issue_acceptance};
    die "missing acquired campaign lease\n" unless ($result->{lock_mode} // '') eq 'verified_inherited_descriptor';
    my $m = $result->{manifest};
    die "wrong counter case\n" unless ref($m) eq 'HASH' && ($m->{schema} // '') eq 'flash.issue10.counter-workload.v1' &&
        ($m->{nv} // 0) == $nv && ($m->{pattern} // '') eq $pattern && ($m->{tail} // -1) == 0 &&
        ($m->{iterations} // 0) == 4 && ($m->{seed} // -1) == 4242;
    die "invalid capture source or GPU identity\n" unless ($m->{expected_source_revision} // '') =~ /\A[0-9a-f]{40}\z/ &&
        ($m->{gpu_uuid} // '') =~ /\AGPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\z/;
    for my $key (qw(started_utc finished_utc)) {
        die "missing capture timestamp\n" unless ($result->{$key} // '') =~ /\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\z/;
    }
    die "capture timestamps reversed\n" if $result->{finished_utc} lt $result->{started_utc};
    my $manifest_record = raw_identity($result->{manifest_identity}, "$dir.manifest.json");
    same(load("$dir.manifest.json"), $m, 'archived workload manifest');
    my %input;
    die "missing input identities\n" unless ref($result->{input_identities}) eq 'ARRAY' && @{$result->{input_identities}};
    for my $id (@{$result->{input_identities}}) {
        die "invalid input identity\n" unless ref($id) eq 'HASH' && defined($id->{path}) &&
            ($id->{sha256} // '') =~ /\A[0-9a-f]{64}\z/ && defined($id->{bytes}) && $id->{bytes} > 0 && !$input{$id->{path}};
        $input{$id->{path}} = {bytes=>$id->{bytes}, sha256=>$id->{sha256}};
    }
    for my $key (qw(binary cubin profiler)) { die "unidentified $key\n" unless $input{$m->{$key}}; }
    my $collector = $result->{collector_identity};
    raw_identity($collector, "$dir/../decode_counter_capture.pl");
    for my $phase (qw(before after)) {
        my $gpu = load("$dir/gpu-$phase.json");
        die "missing successful GPU snapshot\n" unless ($gpu->{exit} // -1) == 0 &&
            ($gpu->{stdout} // '') =~ /<nvidia_smi_log>/ &&
            index($gpu->{stdout}, '<uuid>'.$m->{gpu_uuid}.'</uuid>') >= 0;
    }
    my $apps = load("$dir/compute-apps.json");
    die "failed compute-process inventory\n" unless ($apps->{exit} // -1) == 0;
    for my $line (grep {length} split /\n/, $apps->{stdout}) {
        my $fields = DecodeCounterCapture::csv_fields($line);
        $fields->[0] =~ s/^\s+|\s+$//g;
        die "target GPU had an existing compute process\n" if $fields->[0] eq $m->{gpu_uuid};
    }
    my $expected_group_count = scalar @DecodeCounterCapture::GROUPS;
    my %groups = map {$_->[0] => $_->[1]} @DecodeCounterCapture::GROUPS;
    die "incomplete metric-group matrix\n" unless ref($result->{pairs}) eq 'ARRAY' && @{$result->{pairs}} == $expected_group_count;
    my (%seen, @rows, $reference, $reference_rows, $count);
    for my $pair (@{$result->{pairs}}) {
        my $name = $pair->{group};
        die "unexpected or repeated metric group\n" unless exists($groups{$name}) && !$seen{$name}++;
        die "missing host/VRAM arms\n" unless ref($pair->{arms}) eq 'HASH' && join(',',sort keys %{$pair->{arms}}) eq '0,1';
        my %arms;
        for my $host (0,1) {
            my $stem = "$dir/$name-host$host";
            my $arm = $pair->{arms}{$host};
            same(load("$stem.accepted.json"), $arm, 'accepted arm and campaign record');
            raw_identity($arm->{raw_profiler}, "$stem.nvprof.csv");
            raw_identity($arm->{raw_workload}, "$stem.workload.csv");
            my $recorded_prefix = $arm->{raw_profiler}{path}; $recorded_prefix =~ s/\.nvprof\.csv\z//;
            my $command = load("$stem.command.json");
            validate_command($command,$m,$groups{$name},$host,$recorded_prefix);
            my $workload = DecodeCounterCapture::validate_workload($JSON->decode($command->{stdout}),$m,$host);
            same($workload,$arm->{workload},'raw native output and accepted workload');
            DecodeCounterCapture::matched_workloads($reference,$workload) if $reference;
            $reference //= $workload;
            my $rows = DecodeCounterCapture::workload_rows(DecodeCounterCapture::read_text("$stem.workload.csv"),4);
            same($rows,$arm->{verified_rows},'raw native rows and accepted rows');
            same($rows,$reference_rows,'host/VRAM and metric-group output rows') if $reference_rows;
            $reference_rows //= $rows;
            my $metrics = DecodeCounterCapture::parse_metrics(DecodeCounterCapture::read_text("$stem.nvprof.csv"),$groups{$name},18);
            same($metrics,$arm->{metrics},'raw profiler rows and accepted metrics');
            $arms{$host} = $metrics; $count++;
        }
        for my $metric (@{$groups{$name}}) {
            push @rows, {visible_positions=>4*$nv,pattern=>$pattern,metric=>$metric,unit=>$arms{0}{$metric}{avg}{unit},
                vram=>{map {$_=>$arms{0}{$metric}{$_}{value}} qw(min max avg)},
                host=>{map {$_=>$arms{1}{$metric}{$_}{value}} qw(min max avg)}};
        }
    }
    return {case=>basename($dir),nv=>$nv,pattern=>$pattern,profiled_commands=>$count,metric_rows=>2*@rows,
        started_utc=>$result->{started_utc},finished_utc=>$result->{finished_utc},rows=>\@rows,
        input_identities=>\%input,collector_sha256=>$collector->{sha256},gpu_uuid=>$m->{gpu_uuid},
        source_revision=>$m->{expected_source_revision},manifest_sha256=>$manifest_record->{sha256}};
}
sub replay {
    my ($root) = @_;
    my @wanted = sort map {my $nv=$_; map {"nv$nv-$_"} @PATTERNS} @NV;
    opendir my $dh, $root or die "open matrix: $!\n";
    my @dirs = sort grep {/^nv/ && -d "$root/$_"} readdir($dh);
    closedir $dh;
    same(\@dirs,\@wanted,'complete eight-case directory matrix');
    die "campaign queue did not finish successfully\n" unless DecodeCounterCapture::read_text("$root/queue.exit") eq "0\n";
    my (@cases,$reference);
    for my $nv (@NV) {
        for my $pattern (@PATTERNS) {
            my $r = case_replay("$root/nv$nv-$pattern",$nv,$pattern);
            my $common={map {$_=>$r->{$_}} qw(input_identities collector_sha256 gpu_uuid source_revision)};
            same($common,$reference,'cross-case executable/device identity') if $reference;
            $reference //= $common; push @cases,$r;
        }
    }
    my $commands=0; my $metrics=0; $commands+=$_->{profiled_commands},$metrics+=$_->{metric_rows} for @cases;
    die "wrong matrix coverage\n" unless $commands==64 && $metrics==208;
    return {schema=>'flash.issue10.counter-matrix-replay.v1',status=>'PASS',scope=>'CPU replay of archived component counter matrix; no GPU execution',
        full_issue_acceptance=>JSON::PP::false,cases=>\@cases,profiled_commands=>$commands,metric_rows=>$metrics,
        analyzer_identities=>[map {DecodeCounterCapture::identity($_)}
            (__FILE__,dirname(abs_path(__FILE__)).'/decode_counter_capture.pl')],
        notes=>['Counters include preflight, repeated-row warmups, and finite die framework.',
        'Only changing mode changes selected rows each iteration; clustered/even/random modes retain fixed rows.',
        'No checkpoint-derived layout, full-model throughput, or historical slowdown attribution is claimed.',
        'This replay checks consistency of supplied records; authenticity is anchored to the reviewed Git commit.']};
}
sub main {
    my @args=@_; my ($root,$out);
    GetOptionsFromArray(\@args,'archive=s'=>\$root,'out=s'=>\$out) or die "invalid options\n";
    die "--archive and new --out required\n" unless defined($root) && defined($out) && !@args;
    die "output already exists\n" if -e $out || -l $out;
    my $r=replay($root);
    make_path($out);
    DecodeCounterCapture::save("$out/summary.json",$r);
    my $csv="visible_positions,pattern,metric,unit,vram_min,vram_max,vram_avg,host_min,host_max,host_avg\n";
    for my $c (@{$r->{cases}}) {
        for my $row (@{$c->{rows}}) {
            $csv.=join(',',@$row{qw(visible_positions pattern metric unit)},
                @{$row->{vram}}{qw(min max avg)},@{$row->{host}}{qw(min max avg)})."\n";
        }
    }
    DecodeCounterCapture::write_text("$out/metrics.csv",$csv);
    print "PASS: 8 cases, 64 profiler commands, 208 metric rows; component scope only\n";
    return 0;
}
unless(caller) {exit main(@ARGV);}
1;
