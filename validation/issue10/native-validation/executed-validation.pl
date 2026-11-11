#!/usr/bin/env perl
use strict;
use warnings;
use JSON::PP;
use Digest::SHA;
use POSIX qw(strftime);
use File::Path qw(make_path);
use Cwd qw(abs_path);
my $repo='/home/nate/codex/issue10-worker-qfNDNQcU/tree';
my $out='/home/nate/codex/issue10-worker-qfNDNQcU/validation-at-7d19d09';
die "output exists\n" if -e $out;
make_path($out);
my $json=JSON::PP->new->canonical->pretty;
sub utc {strftime('%Y-%m-%dT%H:%M:%SZ',gmtime)}
sub read_text {open my $f,'<:raw',$_[0] or die "$!";local $/;my $s=<$f>;close $f;return $s}
my @checks;
sub capture {
    my($name,@argv)=@_;my $start=utc();
    my $pid=fork();die "fork: $!" unless defined $pid;
    unless($pid) {
        chdir $repo or die "$!";
        open STDOUT,'>:raw',"$out/$name.log" or die "$!";
        open STDERR,'>&',\*STDOUT or die "$!";
        exec {$argv[0]} @argv;die "exec $argv[0]: $!";
    }
    waitpid($pid,0)==$pid or die "$!";
    my $status=$?;
    my $r={name=>$name,argv=>\@argv,cwd=>$repo,started_utc=>$start,finished_utc=>utc(),
        raw_wait_status=>$status,exit_code=>$status>>8,signal=>$status&127,log=>"$name.log"};
    push @checks,$r;
    die "$name failed, see $out/$name.log\n" if $status;
}
capture('source','git','rev-parse','HEAD');
my $source=read_text("$out/source.log");chomp $source;
die "unexpected source\n" unless $source eq '7d19d09c6ab734847bf737cfa75448161fb092e9';
capture('worktree','git','status','--porcelain=v1');
die "dirty code\n" if length read_text("$out/worktree.log");
capture('host','uname','-a');
capture('perl','perl','-V:version','-V:archname');
capture('driver-syntax','perl','-c','kern/decode_forensics.pl');
capture('archive-syntax','perl','-c','kern/DecodeArchive.pm');
capture('tests','prove','-v','kern/decode_forensics_test.pl');
capture('replay','perl','kern/decode_forensics.pl','--archive','validation/issue10','--out',"$out/replay");
capture('diff-check','git','diff','--check','HEAD^','HEAD');
my $tap=read_text("$out/tests.log");
my($test_count)=$tap=~/Tests=(\d+)/;die "missing test count\n" unless defined $test_count;
my %code;
for my $p(qw(kern/decode_forensics.pl kern/DecodeArchive.pm kern/decode_forensics_test.pl)) {
    open my $f,'<:raw',"$repo/$p" or die "$!";
    $code{$p}=Digest::SHA->new(256)->addfile($f)->hexdigest;
}
my $result={schema=>'flash.issue10.native-validation.v1',status=>'PASS',source_commit=>$source,
    finished_utc=>utc(),host=>'engi',scope=>'CPU-only validation of preserved compact evidence; no GPU inference, raw CSV re-read, or issue-10 completion claim',
    worktree_clean_before_checks=>JSON::PP::true,test_count=>0+$test_count,source_sha256=>\%code,
    checks=>\@checks,replay=>$json->decode(read_text("$out/replay/replay.json"))};
open my $f,'>:raw',"$out/result.json" or die "$!";print {$f} $json->encode($result);close $f or die "$!";
print $json->encode({status=>'PASS',source_commit=>$source,test_count=>0+$test_count,out=>$out});
