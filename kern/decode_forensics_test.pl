#!/usr/bin/env perl
use strict;
use warnings;
use Test::More;
use Storable qw(dclone);
use File::Temp qw(tempdir);
use File::Copy qw(copy);
use FindBin qw($Bin);
use JSON::PP;
use IO::Compress::Gzip qw(gzip $GzipError);
require "$Bin/decode_forensics.pl";
require "$Bin/DecodeArchive.pm";

# CPU-only checks against the preserved compact archive, plus deliberate corruptions.
# These tests make no claim about raw CSV reconstruction or issue-10 GPU acceptance.
my $root="$Bin/../validation/issue10";
my $traced=DecodeArchive::read_json("$root/traced.json.gz");
my $untraced=DecodeArchive::read_json("$root/untraced.json.gz");
my $phase=DecodeArchive::read_json("$root/traced-all-die-phase-totals.json.gz");
my $result=DecodeArchive::validate_data($traced,$untraced,$phase);
is_deeply([@$result{qw(traced_runs untraced_runs die_frames)}],[60,20,360],
    'complete archived matrix has 60 traced, 20 untraced and 360 die records');
cmp_ok($result->{max_recomputed_residual_us},'<=',1e-5,'all die arithmetic partitions reconcile');
for my $pair (['traced_summary','traced-summary.json'],
              ['untraced_summary','untraced-summary.json'],
              ['matched_phase_deltas','matched-phase-deltas.json']) {
    cmp_ok(DecodeArchive::compare($result->{$pair->[0]},DecodeArchive::read_json("$root/$pair->[1]"),$pair->[1]),
        '<=',1e-8,"regenerates independent saved $pair->[1]");
}
is(DecodeForensics::median([5,1,2,4,3]),3,'median is over the five run means');
is(DecodeForensics::p95([reverse 1..640]),608,'p95 uses nearest rank over 640 token intervals');
is_deeply([map {DecodeArchive::selected_count($_)} (0,2046,2047,2048,2050,2051,3099,7999,8000)],
    [1,2047,2048,2049,2051,2048,2048,2048,2049],
    'selected-key count uses zero-based position, four-token blocks, cap and tail');

sub rejects {
    my($name,$call,$expected)=@_;
    my $ok=eval {$call->();1};my $error=$@;
    ok(!$ok && $error=~$expected,$name) or diag("expected $expected; got ".($error||'success'));
}
sub bad_run {
    my($name,$mutation,$expected,$untraced_case)=@_;
    rejects($name,sub {
        my $x=dclone($untraced_case?$untraced:$traced);
        $mutation->($x);
        DecodeArchive::validate_runs($x,$untraced_case?0:1);
    },$expected);
}
my($runs)=DecodeArchive::validate_runs($traced,1);
sub bad_phase {
    my($name,$mutation,$expected)=@_;
    rejects($name,sub {
        my $x=dclone($phase);$mutation->($x);
        DecodeArchive::validate_phases($x,$runs,$traced->{manifest}{configs});
    },$expected);
}

bad_run('rejects missing traced run',sub {pop @{$_[0]{runs}}},qr/wrong number of runs/);
bad_run('rejects missing untraced run',sub {pop @{$_[0]{runs}}},qr/wrong number of runs/,1);
bad_run('rejects duplicated run replacing another pair',sub {$_[0]{runs}[1]=dclone($_[0]{runs}[0])},qr/duplicate run/);
bad_run('rejects unrequested depth',sub {$_[0]{runs}[0]{depth}=9999},qr/invalid run key/);
bad_run('rejects incomplete run',sub {$_[0]{runs}[0]{status}='error'},qr/incomplete run/);
bad_run('rejects a sixth pair',sub {$_[0]{runs}[0]{pair}=5},qr/invalid run key/);
bad_run('rejects different block budget',sub {$_[0]{manifest}{configs}{host}{nblk}=4096},qr/invalid configuration/);
bad_run('rejects unmatched PLE configuration',sub {$_[0]{manifest}{configs}{host}{ple_disk}=0},qr/unmatched configurations/);
bad_run('rejects mismatched source identity',sub {$_[0]{manifest}{configs}{host}{rev}='a'x40},qr/configuration source differs/);
bad_run('rejects truthy string instead of JSON trace flag',sub {$_[0]{traced}='true'},qr/wrong archive schema/);
bad_run('rejects truncated generated IDs',sub {pop @{$_[0]{runs}[0]{generated_ids}}},qr/wrong output dimensions/);
bad_run('rejects changed paired continuation',sub {$_[0]{runs}[0]{generated_ids}[0]++},qr/paired output mismatch/);
bad_run('rejects early stop',sub {$_[0]{runs}[0]{done}{ended}=1},qr/incomplete completion/);
bad_run('rejects incorrect prefix reuse',sub {$_[0]{runs}[0]{done}{resumed}--},qr/incomplete completion/);
bad_run('rejects raw completion inconsistent with parsed fields',sub {$_[0]{runs}[0]{raw_done}=~s/\Adone 129 /done 128 /},qr/raw completion generated differs/);
bad_run('rejects altered interval',sub {$_[0]{runs}[0]{intervals_ns}[0]++},qr/observation interval differs/);
bad_run('rejects repeated arrival timestamp',sub {$_[0]{runs}[0]{client_observed_ns}[1]=$_[0]{runs}[0]{client_observed_ns}[0]},qr/nonpositive observation interval/);
bad_run('rejects incorrect request TTFT',sub {$_[0]{runs}[0]{request_ttft_ms}++},qr/request TTFT differs/);
bad_run('rejects observations outside request',sub {$_[0]{runs}[0]{request_start_ns}=$_[0]{runs}[0]{request_end_ns}+1},qr/observations outside request/);
bad_run('rejects nonfinite timing before tolerance comparison',sub {$_[0]{runs}[0]{client_ms_per_token}='NaN'},qr/invalid number/);
bad_run('rejects stale embedded summary',sub {$_[0]{summary}[0]{median_ms_per_token}++},qr/embedded summary.*differs/);
bad_run('rejects missing larger-context capacity record',sub {pop @{$_[0]{capacity_skips}}},qr/wrong capacity skip coverage/);

bad_phase('rejects missing all-die frame',sub {pop @{$_[0]}},qr/incomplete all-die archive/);
bad_phase('rejects repeated all-die frame',sub {$_[0][1]=dclone($_[0][0])},qr/unknown or repeated phase frame/);
bad_phase('rejects missing die',sub {pop @{$_[0][0]{dies}}},qr/missing dies/);
bad_phase('rejects repeated die',sub {$_[0][0]{dies}[1]=dclone($_[0][0]{dies}[0])},qr/invalid or repeated die/);
bad_phase('rejects truncated trace',sub {$_[0][0]{dies}[1]{metadata}{count}=128},qr/invalid trace metadata/);
bad_phase('rejects mixed requests',sub {$_[0][0]{dies}[1]{metadata}{request}++},qr/mixed request IDs/);
bad_phase('binds complete phase frame to its run projection',sub {$_->{metadata}{request}++ for @{$_[0][0]{dies}}},qr/die0 projection.*request.*differs/);
bad_phase('rejects missing phase component',sub {delete $_[0][0]{dies}[1]{components_us_per_token}{dispatch}},qr/wrong timeline components/);
bad_phase('rejects negative phase time',sub {$_[0][0]{dies}[1]{components_us_per_token}{dispatch}=-1},qr/negative phase time/);
bad_phase('rejects nonfinite phase time',sub {$_[0][0]{dies}[1]{components_us_per_token}{dispatch}='Inf'},qr/invalid number/);
bad_phase('rejects unreconciled duration',sub {$_[0][0]{dies}[1]{device_us_per_token}++},qr/timeline residual differs/);
bad_phase('rejects changed opcode count',sub {$_[0][0]{dies}[1]{op_counts}{idxsel}--},qr/invalid per-token opcode count/);
bad_phase('rejects time attributed to zero executions',sub {$_[0][0]{dies}[1]{op_counts}{idxsel}=0},qr/invalid per-token opcode count/);
bad_phase('rejects fractional execution count',sub {$_[0][0]{dies}[1]{op_counts}{idxsel}+=0.5},qr/invalid integer/);
bad_phase('rejects opcode/component disagreement',sub {$_[0][0]{dies}[1]{op_mean_us}{idxsel}{dispatch}++},qr/opcode sum dispatch differs/);
bad_phase('rejects wrong selected-key count',sub {$_[0][0]{dies}[1]{selected_keys}[0][1]++},qr/wrong selected-key count/);
bad_phase('rejects shifted interior selected-key position',sub {$_[0][0]{dies}[1]{selected_keys}[64][0]++},qr/wrong selected-key position/);
bad_phase('rejects nonfinite recorded residual',sub {$_[0][0]{dies}[1]{reconciliation_residual_us}='NaN'},qr/invalid number/);

for my $bad (undef,'NaN','Inf','-Inf',JSON::PP::true) {
    rejects('rejects undefined/nonfinite/boolean numeric input',sub {DecodeArchive::number($bad,'test')},qr/invalid number/);
}
rejects('rejects fractional integer',sub {DecodeArchive::number(1.5,'test',1)},qr/invalid integer/);
rejects('rejects different reference schema',sub {DecodeArchive::compare({a=>1},{b=>1},'reference')},qr/different keys/);

# A small copied archive verifies the identity checks and publication behavior.
my $tmp=tempdir('decode-archive-test-XXXXXXXX',TMPDIR=>1,CLEANUP=>1);
my @required=qw(traced.json.gz untraced.json.gz traced-all-die-phase-totals.json.gz
    traced-summary.json untraced-summary.json matched-phase-deltas.json);
my %ids;
for my $p (@required) {
    copy("$root/$p","$tmp/$p") or die "copy $p: $!";
    $ids{$p}={bytes=>0+(-s "$tmp/$p"),sha256=>DecodeForensics::sha("$tmp/$p")};
}
DecodeForensics::save("$tmp/IDENTITIES.json",\%ids);
is(DecodeArchive::verify_identities($tmp)->{files},6,'checks all required artifact identities');
{
    my $changed=dclone(\%ids);delete $changed->{'traced-summary.json'};
    DecodeForensics::save("$tmp/IDENTITIES.json",$changed);
    rejects('requires reference summary identity',sub {DecodeArchive::verify_identities($tmp)},qr/identity missing/);
}
{
    my $changed=dclone(\%ids);$changed->{'../outside'}=$ids{'traced-summary.json'};
    DecodeForensics::save("$tmp/IDENTITIES.json",$changed);
    rejects('rejects parent traversal in identity manifest',sub {DecodeArchive::verify_identities($tmp)},qr/unsafe identity path/);
}
DecodeForensics::save("$tmp/IDENTITIES.json",\%ids);
{
    my $changed=dclone(\%ids);
    @$changed{qw(traced-summary.json untraced-summary.json)}=@$changed{qw(untraced-summary.json traced-summary.json)};
    DecodeForensics::save("$tmp/IDENTITIES.json",$changed);
    rejects('binds identities to the required logical filenames',sub {DecodeArchive::verify_identities($tmp)},qr/artifact (?:size|hash) mismatch/);
    DecodeForensics::save("$tmp/IDENTITIES.json",\%ids);
}
{
    open my $f,'+<:raw',"$tmp/traced-summary.json" or die "$!";
    print {$f} '!' or die "$!";close $f or die "$!";
    rejects('rejects changed bytes even at identical size',sub {DecodeArchive::verify_identities($tmp)},qr/artifact hash mismatch/);
    rejects('corrupt archive cannot publish a PASS directory',sub {DecodeArchive::replay($tmp,"$tmp/rejected")},qr/artifact hash mismatch/);
    ok(!-e "$tmp/rejected",'failed validation leaves no output directory');
    copy("$root/traced-summary.json","$tmp/traced-summary.json") or die "$!";
}
{
    my $stdout='';
    open my $capture,'>',\$stdout or die "$!";
    local *STDOUT=$capture;
    is(DecodeForensics::main('--archive',$tmp,'--out',"$tmp/replay"),0,'native CLI archive mode completes');
}
my $record=DecodeArchive::read_json("$tmp/replay/replay.json");
is($record->{status},'PASS','successful replay publishes its scoped status');
is($record->{identity_manifest_sha256},DecodeForensics::sha("$tmp/IDENTITIES.json"),'replay binds the input identity manifest');
is_deeply([sort keys %{$record->{analyzer_sha256}}],[qw(kern/DecodeArchive.pm kern/decode_forensics.pl)],'replay identifies both source files');
for my $p (qw(traced-summary.json untraced-summary.json matched-phase-deltas.json)) {
    cmp_ok(DecodeArchive::compare(DecodeArchive::read_json("$tmp/replay/$p"),DecodeArchive::read_json("$root/$p"),$p),
        '<=',1e-8,"CLI output agrees with saved $p");
}
rejects('refuses overwriting an existing replay',sub {DecodeForensics::main('--archive',$tmp,'--out',"$tmp/replay")},qr/output directory already exists/);
rejects('refuses mixed archive and raw capture inputs',sub {DecodeForensics::main('--archive',$tmp,'--traced',$tmp,'--out',"$tmp/mixed")},qr/excludes/);
{
    DecodeForensics::save("$tmp/truncated.gz",{valid=>1});
    truncate "$tmp/truncated.gz",(-s "$tmp/truncated.gz")-4 or die "$!";
    rejects('rejects gzip without a complete trailer',sub {DecodeArchive::read_json("$tmp/truncated.gz")},qr/gunzip/);
    DecodeForensics::save("$tmp/trailing.gz",{valid=>1});
    open my $f,'>>:raw',"$tmp/trailing.gz" or die "$!";
    print {$f} 'unexpected' or die "$!";close $f or die "$!";
    rejects('rejects trailing bytes after gzip',sub {DecodeArchive::read_json("$tmp/trailing.gz")},qr/trailing data after gzip/);
    my $payload='{} trailing garbage';
    gzip(\$payload=>"$tmp/invalid-json.gz") or die "$GzipError";
    rejects('rejects trailing data after a valid JSON prefix',sub {DecodeArchive::read_json("$tmp/invalid-json.gz")},qr/garbage after JSON/);
}
{
    # Consistently alter client and die-0 timing. The independent, preserved
    # summaries must still reject it even after all internal projections agree.
    my $t=dclone($traced);my $p=dclone($phase);my $r=$t->{runs}[0];
    my($frame)=grep {DecodeArchive::key($_) eq DecodeArchive::key($r)} @$p;
    my($die0)=grep {$_->{die}==0} @{$frame->{dies}};
    $r->{client_observed_ns}[-1]+=1e6;$r->{intervals_ns}[-1]+=1e6;
    $r->{client_decode_ms}++;$r->{client_ms_per_token}+=1/128;
    $r->{request_end_ns}+=1e6;$r->{request_completion_ms}++;
    $r->{client_minus_engine_interval_ms}++;$r->{engine_minus_gpu0_interval_ms}--;
    $r->{device_publication_decode_ms_by_die}[0]++;
    $die0->{device_us_per_token}+=1000/128;
    $die0->{components_us_per_token}{between_token}+=1000/128;
    $r->{die0_timeline}=dclone($die0);
    $t->{summary}=DecodeArchive::summary($t->{runs});
    ok(eval {DecodeArchive::validate_data($t,$untraced,$p);1},'coordinated corruption has internally agreeing projections') or diag($@);
    DecodeForensics::save("$tmp/traced.json.gz",$t);
    DecodeForensics::save("$tmp/traced-all-die-phase-totals.json.gz",$p);
    for my $file(qw(traced.json.gz traced-all-die-phase-totals.json.gz)) {
        $ids{$file}={bytes=>0+(-s "$tmp/$file"),sha256=>DecodeForensics::sha("$tmp/$file")};
    }
    DecodeForensics::save("$tmp/IDENTITIES.json",\%ids);
    rejects('independent reference rejects internally consistent altered timings',sub {DecodeArchive::replay($tmp,"$tmp/coordinated")},qr/traced-summary\.json.*differs/);
    ok(!-e "$tmp/coordinated",'reference mismatch publishes no PASS directory');
}
done_testing();
