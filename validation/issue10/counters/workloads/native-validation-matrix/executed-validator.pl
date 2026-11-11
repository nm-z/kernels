#!/usr/bin/env perl
use strict;
use warnings;
use Cwd qw(abs_path);
use JSON::PP;
use File::Path qw(make_path);
require './kern/decode_counter_capture.pl';
my $out=shift or die "validation output required\n";
die "output exists\n" if -e $out;
make_path($out);
my @code=qw(kern/decode_counter_capture.pl kern/decode_counter_capture_test.pl kern/decode_counter_matrix.pl kern/decode_counter_matrix_test.pl);
my @checks=(
 ['collector-syntax',['perl','-c',$code[0]],30],
 ['collector-test-syntax',['perl','-c',$code[1]],30],
 ['matrix-syntax',['perl','-c',$code[2]],30],
 ['matrix-test-syntax',['perl','-c',$code[3]],30],
 ['tests',['prove','-v',$code[1],$code[3],'kern/decode_forensics_test.pl'],180],
 ['matrix-replay',['perl',$code[2],'--archive','validation/issue10/counters/workloads/matrix-3100-8000','--out',"$out/matrix-replay"],90],
 ['legacy-replay',['perl','kern/decode_forensics.pl','--archive','validation/issue10','--out',"$out/legacy-replay"],90],
 ['staged-whitespace',['git','diff','--cached','--check'],30],
 ['campaign-shell-syntax',['bash','-n','validation/issue10/counters/workloads/matrix-3100-8000/queue-counters.sh'],30],
 ['readiness-shell-syntax',['bash','-n','validation/issue10/counters/workloads/full-model-readiness-20261018/executed-readiness.sh'],30],
);
my $diff=DecodeCounterCapture::run_command(['git','diff','--cached','--binary','--',@code],30);
die "code diff failed\n" if $diff->{exit};
DecodeCounterCapture::write_text("$out/tested-code.patch",$diff->{stdout});
my $r={schema=>'flash.issue10.resume-validation.v1',status=>'INCOMPLETE',
 scope=>'CPU syntax, regression tests, and independent replay of saved component and legacy model captures; no GPU work by this driver',
 hostname=>DecodeCounterCapture::run_command(['hostname'],10)->{stdout},cwd=>abs_path('.'),
 base_commit=>DecodeCounterCapture::run_command(['git','rev-parse','HEAD'],10)->{stdout},
 worktree_status=>DecodeCounterCapture::run_command(['git','status','--porcelain'],10)->{stdout},
 source_identities=>[map {DecodeCounterCapture::identity($_)} @code],
 tested_code_patch=>DecodeCounterCapture::identity("$out/tested-code.patch"),
 driver_identity=>DecodeCounterCapture::identity(__FILE__),checks=>[]};
for my $c (@checks) {
 my $run=DecodeCounterCapture::run_command($c->[1],$c->[2]);
 DecodeCounterCapture::write_text("$out/$c->[0].log",$run->{stdout}.$run->{stderr});
 push @{$r->{checks}},{name=>$c->[0],%$run};
 print "$c->[0] exit=$run->{exit}\n";
 if($run->{exit}) {$r->{status}='FAIL';DecodeCounterCapture::save("$out/result.json",$r);die "failed $c->[0]\n";}
 if($c->[0] eq 'tests') {
  die "unexpected test coverage\n" unless $run->{stdout}=~/Files=3, Tests=192,/;
  $r->{test_count}=192;$r->{collector_tests}=67;$r->{matrix_tests}=48;$r->{legacy_archive_tests}=77;
 }
}
$r->{status}='PASS';
DecodeCounterCapture::save("$out/result.json",$r);
print "PASS: 192 CPU tests and both complete archive replays\n";
