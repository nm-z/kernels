#!/usr/bin/env perl
use strict;
use warnings;
use FindBin qw($Bin);
use File::Temp qw(tempdir);
use File::Path qw(make_path remove_tree);
use JSON::PP;
use Storable qw(dclone);
use Test::More;
require "$Bin/decode_counter_matrix.pl";
my $source="$Bin/../validation/issue10/counters/workloads/matrix-3100-8000";
BAIL_OUT('committed component matrix is missing') unless -d $source;
my $result=DecodeCounterMatrix::replay($source);
is($result->{profiled_commands},64,'all 64 real profiler commands replay');
is($result->{metric_rows},208,'all 208 real metric rows replay');
is(scalar(@{$result->{cases}}),8,'both contexts and all four layouts present');
ok(!$result->{full_issue_acceptance},'component PASS never claims full-model acceptance');
my @host_reads;
my @vram_reads;
for my $case (@{$result->{cases}}) {
    my ($r)=grep {$_->{metric} eq 'sysmem_read_bytes'} @{$case->{rows}};
    push @host_reads,$r->{host}{avg}; push @vram_reads,$r->{vram}{avg};
}
is_deeply(\@host_reads,[(2359296)x8],'reproduces all observed host system-memory read averages');
is_deeply(\@vram_reads,[(0)x8],'reproduces all observed VRAM system-memory read averages');
my $tmp=tempdir(CLEANUP=>1);
my $root="$tmp/matrix";
system('cp','-a','--reflink=auto','--',$source,$root)==0 or BAIL_OUT('cannot copy test fixture');
my $case='nv775-clustered';
my $status="$case/status.json";
sub rejects {
    my ($name,$call,$pattern)=@_;
    my $ok=eval {$call->();1}; my $error=$@;
    ok(!$ok && $error=~$pattern,$name) or diag($error || 'unexpected success');
}
sub bad_json {
    my ($name,$relative,$edit,$pattern)=@_;
    my $path="$root/$relative";
    my $original=DecodeCounterCapture::read_text($path);
    my $data=JSON::PP->new->decode($original);
    $edit->($data);
    DecodeCounterCapture::save($path,$data);
    rejects($name,sub {DecodeCounterMatrix::replay($root)},$pattern);
    DecodeCounterCapture::write_text($path,$original);
}
sub bad_text {
    my ($name,$relative,$edit,$pattern)=@_;
    my $path="$root/$relative";
    my $original=DecodeCounterCapture::read_text($path);
    my $changed=$original; $edit->($changed);
    DecodeCounterCapture::write_text($path,$changed);
    rejects($name,sub {DecodeCounterMatrix::replay($root)},$pattern);
    DecodeCounterCapture::write_text($path,$original);
}
bad_text('failed queue is not accepted','queue.exit',sub {$_[0]="73\n"},qr/queue did not finish/);
rename "$root/nv2000-changing","$root/temporarily-missing" or die $!;
rejects('missing case rejected',sub {DecodeCounterMatrix::replay($root)},qr/eight-case/);
rename "$root/temporarily-missing","$root/nv2000-changing" or die $!;
make_path("$root/nv999-extra");
rejects('extra counter case rejected',sub {DecodeCounterMatrix::replay($root)},qr/eight-case/);
remove_tree("$root/nv999-extra");
for my $c (
    ['failed capture','status','FAIL',qr/completed component/],
    ['full issue acceptance flag','full_issue_acceptance',JSON::PP::true,qr/completed component/],
    ['unverified queue lease','lock_mode','owned_lock',qr/campaign lease/],
    ['bad capture timestamp','started_utc','yesterday',qr/timestamp/],
    ['reversed capture timestamps','finished_utc','2000-01-01T00:00:00Z',qr/reversed/],
) {
    bad_json($c->[0],$status,sub {$_[0]{$c->[1]}=$c->[2]},$c->[3]);
}
for my $c (
    ['wrong context','nv',2000],['wrong layout','pattern','even'],
    ['wrong tail','tail',1],['wrong iterations','iterations',5],['wrong seed','seed',42]
) {
    bad_json($c->[0],$status,sub {$_[0]{manifest}{$c->[1]}=$c->[2]},qr/wrong counter case/);
}
bad_json('invalid source revision',$status,sub {$_[0]{manifest}{expected_source_revision}='unknown'},qr/source or GPU/);
bad_json('invalid GPU UUID',$status,sub {$_[0]{manifest}{gpu_uuid}='6'},qr/source or GPU/);
bad_json('missing metric group',$status,sub {pop @{$_[0]{pairs}}},qr/metric-group/);
bad_json('repeated metric group',$status,sub {$_[0]{pairs}[1]=dclone($_[0]{pairs}[0])},qr/repeated metric/);
bad_json('unknown metric group',$status,sub {$_[0]{pairs}[0]{group}='imagined'},qr/unexpected/);
bad_json('missing host arm',$status,sub {delete $_[0]{pairs}[0]{arms}{1}},qr/missing host/);
bad_json('unidentified executable',$status,sub {shift @{$_[0]{input_identities}}},qr/unidentified binary/);
bad_json('invalid artifact identity',$status,sub {$_[0]{input_identities}[0]{sha256}='not-a-hash'},qr/input identity/);
bad_json('different raw manifest',"$case.manifest.json",sub {$_[0]{seed}=0},qr/raw artifact identity/);
bad_json('failed hardware snapshot',"$case/gpu-before.json",sub {$_[0]{exit}=1},qr/GPU snapshot/);
bad_json('different hardware UUID',"$case/gpu-after.json",sub {$_[0]{stdout}=~s/GPU-431371bd-b482-96b2-4b70-04f83c6d075b/GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/g},qr/GPU snapshot/);
bad_json('failed process inventory',"$case/compute-apps.json",sub {$_[0]{exit}=1},qr/process inventory/);
bad_json('occupied target GPU',"$case/compute-apps.json",sub {$_[0]{stdout}.="GPU-431371bd-b482-96b2-4b70-04f83c6d075b, 123, other, 100\n"},qr/existing compute/);
my $cmd="$case/traffic-host0.command.json";
bad_json('profile command failure',$cmd,sub {$_[0]{exit}=124},qr/did not exit successfully/);
bad_json('profile command wrong device',$cmd,sub {for(@{$_[0]{argv}}){s/^CUDA_VISIBLE_DEVICES=.*/CUDA_VISIBLE_DEVICES=other/}},qr/GPU environment/);
bad_json('conflicting duplicate GPU environment',$cmd,sub {unshift @{$_[0]{argv}},'CUDA_VISIBLE_DEVICES=other'},qr/GPU environment/);
bad_json('profile command wrong kernel',$cmd,sub {my $a=$_[0]{argv}; for my $i(0..$#$a-1){$a->[$i+1]='wrong' if $a->[$i] eq '--kernels'}},qr/arguments differs/);
bad_json('profile command missing identity',$cmd,sub {for(@{$_[0]{argv}}){$_='another-profiler' if m{/bin/nvprof$}}},qr/profiler identity/);
bad_json('raw native stdout differs',$cmd,sub {my $w=JSON::PP->new->decode($_[0]{stdout});$w->{cache_fnv1a64}='a'x16;$_[0]{stdout}=JSON::PP->new->encode($w)},qr/accepted workload differs/);
bad_json('accepted metric record is not independent truth',"$case/traffic-host0.accepted.json",sub {$_[0]{metrics}{sysmem_read_bytes}{avg}{value}=100},qr/campaign record differs/);
bad_text('truncated raw profiler rejected',"$case/traffic-host0.nvprof.csv",sub {chop $_[0]},qr/raw artifact identity/);
bad_text('altered raw workload rejected',"$case/traffic-host0.workload.csv",sub {$_[0]=~s/iteration/Iteration/},qr/raw artifact identity/);
bad_json('cross-case binary mismatch','nv2000-changing/status.json',sub {$_[0]{input_identities}[0]{sha256}='a'x64},qr/cross-case executable/);
# A locally coordinated checksum change must still reach the raw parser.
{
    my $raw="$root/$case/traffic-host0.nvprof.csv";
    my $accepted="$root/$case/traffic-host0.accepted.json";
    my $path="$root/$status";
    my %old=map {$_=>DecodeCounterCapture::read_text($_)} ($raw,$accepted,$path);
    my $bad=$old{$raw}; $bad=~s/"sysmem_read_bytes"/"invented_metric"/;
    DecodeCounterCapture::write_text($raw,$bad);
    my $id=DecodeCounterCapture::identity($raw);
    my $a=JSON::PP->new->decode($old{$accepted});
    my $s=JSON::PP->new->decode($old{$path});
    for my $record ($a->{raw_profiler},$s->{pairs}[0]{arms}{0}{raw_profiler}) {
        $record->{sha256}=$id->{sha256};$record->{bytes}=$id->{bytes};
    }
    DecodeCounterCapture::save($accepted,$a);DecodeCounterCapture::save($path,$s);
    rejects('coordinated raw-hash mutation still fails metric schema',sub {DecodeCounterMatrix::replay($root)},qr/unexpected\/duplicate metric/);
    DecodeCounterCapture::write_text($_,$old{$_}) for keys %old;
}
my $out="$tmp/replay";
is(DecodeCounterMatrix::main('--archive',$root,'--out',$out),0,'CLI regenerates successful summary');
my @csv=split /\n/,DecodeCounterCapture::read_text("$out/metrics.csv");
is(scalar(@csv),105,'wide CSV includes all 104 matched metric rows and header');
rejects('never overwrite prior replay output',sub {DecodeCounterMatrix::main('--archive',$root,'--out',$out)},qr/output already exists/);
my $bad_exit=DecodeCounterCapture::read_text("$root/queue.exit");
DecodeCounterCapture::write_text("$root/queue.exit","73\n");
rejects('bad campaign publishes no PASS output',sub {DecodeCounterMatrix::main('--archive',$root,'--out',"$tmp/rejected")},qr/queue did not finish/);
ok(!-e "$tmp/rejected",'failed replay leaves no success directory');
DecodeCounterCapture::write_text("$root/queue.exit",$bad_exit);
done_testing;
