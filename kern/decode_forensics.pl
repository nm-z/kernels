#!/usr/bin/env perl
package DecodeForensics;
use strict;
use warnings;
use Digest::SHA;
use File::Find;
use File::Basename qw(dirname);
use Cwd qw(abs_path);
use File::Path qw(make_path);
use Getopt::Long qw(GetOptionsFromArray);
use IO::Compress::Gzip qw($GzipError);
use JSON::PP;
use POSIX qw(ceil);

# Analyze saved evidence. GPU clock epochs are local to each die.
our @PARTS=qw(dispatch exchange_wait body_excluding_ple_wait barrier_send ple_wait);
our @ATTENTION=qw(attnqkq idxqk idxscore idxsel kvgather attnpartn attnmerge);
our @ONCE=qw(attnqkq idxqk idxscore kvgather attnpartn attnmerge);
our $JSON=JSON::PP->new->canonical->allow_nonref;
our $PRETTY=JSON::PP->new->canonical->pretty->allow_nonref;

sub read_text {
    my($p)=@_;open my $f,'<:raw',$p or die "open $p: $!\n";
    local $/;my $s=<$f>;close $f or die "close $p: $!\n";
    return defined($s)?$s:'';
}
sub load {return $JSON->decode(read_text($_[0]));}
sub sha {
    my($p)=@_;open my $f,'<:raw',$p or die "open $p: $!\n";
    my $h=Digest::SHA->new(256);$h->addfile($f);
    close $f or die "close $p: $!\n";return $h->hexdigest;
}
sub save {
    my($p,$v)=@_;my $s=$PRETTY->encode($v);
    if($p=~/\.gz\z/){
        my $f=IO::Compress::Gzip->new($p,Name=>'',Time=>0)
          or die "gzip $p: $GzipError\n";
        print {$f} $s or die "write gzip $p: $GzipError\n";
        $f->close or die "close gzip $p: $GzipError\n";
    }else{
        open my $f,'>:raw',$p or die "open $p: $!\n";
        print {$f} $s or die "write $p: $!\n";
        close $f or die "close $p: $!\n";
    }
}
sub total {my $n=0;$n+=$_ for @{$_[0]};return $n;}
sub same {return $JSON->encode($_[0]) eq $JSON->encode($_[1]);}
sub median {
    my($v)=@_;die "empty median input\n" unless @$v;
    my @s=sort {$a<=>$b} @$v;my $m=int(@s/2);
    return @s%2?$s[$m]:($s[$m-1]+$s[$m])/2;
}
sub p95 {
    my($v)=@_;die "empty percentile input\n" unless @$v;
    my @s=sort {$a<=>$b} @$v;return $s[ceil(.95*@s)-1];
}
sub timeline {
    my($s)=@_;my $t=$s->{tokens};my $n=@$t-1;
    die "expected 129 publication events\n" unless $n==128;
    my %parts;
    for my $k(@PARTS){
        my @values;
        for my $op(values %{$s->{op_mean_us}}){
            die "missing timeline component $k\n" unless exists $op->{$k};
            push @values,$op->{$k};
        }
        $parts{$k}=total(\@values);
    }
    $parts{between_phase}=total([
      map {$t->[$_]{between_phase_ns}} 1..$#$t])/ $n /1000;
    $parts{publication_footer}=total([
      map {$t->[$_]{publication_footer_ns}} 1..$#$t])/ $n /1000;
    $parts{between_token}=total([
      map {$t->[$_]{start_ns}-$t->[$_-1]{publication_ns}} 1..$#$t])/ $n /1000;
    my $exact=($t->[-1]{publication_ns}-$t->[0]{publication_ns})/$n/1000;
    my $residual=$exact-total([values %parts]);
    die "unreconciled timeline $residual us\n" if abs($residual)>1e-5;
    return {components_us_per_token=>\%parts,device_us_per_token=>$exact,
      reconciliation_residual_us=>$residual};
}
sub process {
    my($root,$out,$tag,$traced)=@_;
    my $manifest=load("$root/manifest.json");
    my $results=load("$root/results.json");
    my $status=load("$root/status.json");
    my $exit_status=load("$root/exit.json");
    die "capture did not finish successfully\n" unless
      ($status->{status}//'') eq 'PASS' &&
      defined($exit_status->{exit}) && $exit_status->{exit}==0;
    die "unexpected repetition or output count\n" unless
      $manifest->{pairs}==5 && $manifest->{outputs}==129;
    my $configs=$manifest->{configs};
    die "missing paired KV placements\n" unless ref($configs) eq 'HASH' &&
      join(',',sort keys %$configs) eq 'host,vram' &&
      $configs->{host}{kv_host} eq '1' && $configs->{vram}{kv_host} eq '0';
    my %host=map {$_=>$configs->{host}{$_}}
      grep {$_ ne 'kv_host'} keys %{$configs->{host}};
    my %vram=map {$_=>$configs->{vram}{$_}}
      grep {$_ ne 'kv_host'} keys %{$configs->{vram}};
    die "unmatched realized configurations\n" unless same(\%host,\%vram);
    for my $c(values %$configs){
        die "unexpected model configuration\n" unless
          $c->{nblk}==512 && $c->{layers} eq '48/48';
    }
    my(%inputs,%groups,%seen);my(@runs,@phase,@skips);
    for my $row(@$results){
        if($row->{status} ne 'complete'){
            die "unexpected rejected sample\n" unless
              $row->{status} eq 'capacity_skip' &&
              $row->{depth}+128>$row->{capacity};
            push @skips,$row;next;
        }
        my($depth,$pair,$place)=@$row{qw(depth pair placement)};
        my $key=join("\0",$depth,$pair,$place);
        die "duplicate or invalid pair\n" if $seen{$key}++ ||
          !defined($pair) || $pair<0 || $pair>4 || int($pair)!=$pair ||
          !exists($configs->{$place});
        my $observed=$row->{client_observed_ns};
        my @intervals=map {$observed->[$_]-$observed->[$_-1]} 1..$#$observed;
        die "client intervals do not match raw observations\n" unless
          @$observed==129 && same($row->{intervals_ns},\@intervals);
        die "invalid client decode duration\n" if
          (grep {$_<=0} @{$row->{intervals_ns}}) ||
          abs(total($row->{intervals_ns})/1e6-$row->{client_decode_ms})>1e-9;
        die "wrong completion dimensions\n" unless
          $row->{done}{generated}==129 && $row->{done}{prompt}==$depth;
        die "invalid decode sample\n" unless $row->{decode_intervals}==128 &&
          @{$row->{generated_ids}}==129 && !$row->{done}{ended};
        die "missing prefix reuse\n" unless $row->{done}{resumed}==$depth-1;
        my $input_key=join("\0",$depth,$pair);
        die "paired continuations differ\n" if exists($inputs{$input_key}) &&
          !same($inputs{$input_key},$row->{generated_ids});
        $inputs{$input_key}=$row->{generated_ids};
        my %short=map {$_=>$row->{$_}}
          grep {$_ ne 'request' && $_ ne 'gpu_after' && $_ ne 'raw'} keys %$row;
        $short{raw_done}=$row->{raw}[-1];
        $short{gpu_after}=$row->{gpu_after};
        $short{client_ms_per_token}=$row->{client_decode_ms}/128;
        if($traced){
            my @dies;
            for my $die(0..5){
                my $p="$root/depth-$depth-pair-$pair-$place-die-$die".
                  '.csv.gz.summary.json';
                my $s=load($p);my $r=timeline($s);
                @$r{qw(die op_mean_us op_counts selected_keys metadata source_summary_sha256)}=
                  ($die,$s->{op_mean_us},$s->{op_counts},$s->{selected_keys},
                   $s->{metadata},sha($p));
                push @dies,$r;
            }
            for my $op(@ONCE){
                die "wrong executed opcode count $op\n" unless
                  total([map {$_->{op_counts}{$op}//0} @dies])==12*128;
            }
            die "wrong selection pass count\n" unless
              total([map {$_->{op_counts}{idxsel}//0} @dies])==12*128*5;
            $short{die0_timeline}=$dies[0];
            my %work;
            for my $op(@ATTENTION){
                $work{$op}=total([
                  map {total([values %{$_->{op_mean_us}{$op}//{}}])} @dies]);
            }
            $short{distinct_attention_layer_work_us}=\%work;
            $short{client_minus_device0_us_per_token}=
              1000*$short{client_ms_per_token}-$dies[0]{device_us_per_token};
            push @phase,{depth=>$depth,pair=>$pair,placement=>$place,dies=>\@dies};
        }
        my $short=\%short;
        push @runs,$short;push @{$groups{$depth}{$place}},$short;
    }
    die "capture count does not match results\n" unless $status->{runs}==@runs;
    my @summary;
    for my $depth(sort {$a<=>$b} keys %groups){
        for my $place(sort keys %{$groups{$depth}}){
            my $rows=$groups{$depth}{$place};
            my $other=$place eq 'host'?'vram':'host';
            die "incomplete placement pairs\n" unless @$rows==5 &&
              exists($groups{$depth}{$other}) && @{$groups{$depth}{$other}}==5;
            my @vals=map {$_->{client_ms_per_token}} @$rows;
            my @raw=map {map {$_/1e6} @{$_->{intervals_ns}}} @$rows;
            my $med=median(\@vals);
            push @summary,{depth=>0+$depth,placement=>$place,runs=>scalar(@$rows),
              median_ms_per_token=>$med,median_tok_s=>1000/$med,
              p95_interval_ms=>p95(\@raw),per_run_ms_per_token=>\@vals};
        }
    }
    my $result={schema=>'flash-sass.decode-forensics.v1',tag=>$tag,
      traced=>$traced?JSON::PP::true:JSON::PP::false,
      manifest=>$manifest,summary=>\@summary,capacity_skips=>\@skips,runs=>\@runs,
      note=>'Full model forward on fixed synthetic token IDs. No language/task quality claim. GPU epochs are local; do not sum complete die timelines.'};
    save("$out/$tag.json.gz",$result);
    save("$out/$tag-all-die-phase-totals.json.gz",\@phase) if $traced;
    save("$out/$tag-summary.json",\@summary);
    return $result;
}
sub output_identities {
    my($out)=@_;my @files;
    find({no_chdir=>1,wanted=>sub {
        my $p=$File::Find::name;
        push @files,$p if -f $p && $p ne "$out/IDENTITIES.json";
    }},$out);
    my %identities;
    for my $p(sort @files){
        my $relative=$p;$relative=~s{\A\Q$out\E/}{};
        my @s=stat($p);die "stat $p: $!\n" unless @s;
        $identities{$relative}={bytes=>$s[7],sha256=>sha($p)};
    }
    save("$out/IDENTITIES.json",\%identities);
}
sub main {
    my @args=@_;my($traced,$untraced,$out,$archive);
    GetOptionsFromArray(\@args,'traced=s'=>\$traced,
      'untraced=s'=>\$untraced,'out=s'=>\$out,'archive=s'=>\$archive) or die "invalid options\n";
    die "unexpected arguments: @args\n" if @args;
    if(defined $archive){
        die "--archive requires --out and excludes --traced/--untraced\n"
          unless defined($out)&&!defined($traced)&&!defined($untraced);
        require(dirname(abs_path(__FILE__)).'/DecodeArchive.pm');
        return DecodeArchive::replay($archive,$out);
    }
    die "--traced and --out are required\n" unless defined($traced)&&defined($out);
    $out=~s{/+\z}{} unless $out eq '/';
    die "output directory already exists: $out\n" if -e $out;
    make_path($out);
    my $trace=process($traced,$out,'traced',1);
    process($untraced,$out,'untraced',0) if defined $untraced;
    # Pair 2 illustrates the matched change; all five pairs remain published.
    my %rows;
    for my $r(@{$trace->{runs}}){
        $rows{join("\0",@$r{qw(depth placement pair)})}=$r;
    }
    my %witness;
    for my $place(qw(host vram)){
        my $s=$rows{join("\0",3100,$place,2)};
        my $e=$rows{join("\0",8000,$place,2)};
        die "missing illustrative pair-2 witness for $place\n" unless $s&&$e;
        my $a=$s->{die0_timeline}{components_us_per_token};
        my $b=$e->{die0_timeline}{components_us_per_token};
        my %timeline_delta=map {$_=>$b->{$_}-$a->{$_}} keys %$a;
        my %attention_delta=map {
            $_=>$e->{distinct_attention_layer_work_us}{$_}-
                $s->{distinct_attention_layer_work_us}{$_}
        } keys %{$s->{distinct_attention_layer_work_us}};
        $witness{$place}={pair=>2,
          client_delta_us_per_token=>1000*($e->{client_ms_per_token}-$s->{client_ms_per_token}),
          die0_delta_us_per_token=>$e->{die0_timeline}{device_us_per_token}-
            $s->{die0_timeline}{device_us_per_token},
          timeline_delta_us_per_token=>\%timeline_delta,
          distinct_attention_work_delta_us=>\%attention_delta};
    }
    save("$out/matched-phase-deltas.json",\%witness);
    output_identities($out);
    print $PRETTY->encode({summary=>$trace->{summary},witness=>\%witness});
    return 0;
}
unless(caller){exit main(@ARGV);}
1;
