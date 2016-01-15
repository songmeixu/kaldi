# function：
# author  ：songmeixu
# date    ：

# 参数1：输入
# 参数2：输出

# CMD：

use 5.010;
use utf8;
use Encode;
use Fatal qw/open opendir/;

open MDL, "<:encoding(utf8)", "$ARGV[0]";
open PHN, "<:encoding(utf8)", "$ARGV[1]";
open TRANS, "<:encoding(utf8)", "$ARGV[2]";
open CTX, "<:encoding(utf8)", "$ARGV[3]";
open MMF, ">:encoding(utf8)", "$ARGV[4]";
binmode(STDOUT, ":encoding(gbk)");
binmode(STDERR, ":encoding(gbk)");

# load phones.txt into int2sym
while(my $line = <PHN>) {
  chomp($line);
  my @A = split /\s+/, $line;
  if (@A != 2)
    die "Bad line in symtab file $_: line $.\n";
  if(defined $sym2int{$A[0]}) {
    die "Multiply defined symbol $A[0]";
  }
  $sym2int{$A[0]} = $A[1];
  $int2sym{$A[1]} = $A[0];
}

#load transition prob
while (my $line = <TRANS>) {
  chomp($line);
  my @A = split /\s+/, $line;
  if ($line =~ /Transition-state (\d+): phone = (\S+) hmm-state = (\d+) pdf = (\d+)/) {
    $phone = $2;
    $hmm_state = $3;
    $pdf_id = $4;
  } elsif ($line =~ /Transition-id = (\d+) p = (\S+) count of pdf = (\S+) \[self-loop\]/) {
    $phone_state_pdf_trans{$2}{$3}{$4} = $2;
  } elseif ($line =~ /Transition-id = (\d+) p = (\S+) count of pdf = (\S+) \[0 -> 1\]/) {

  }
}

while (my $line = <IN>) {
  chomp($line);
  my @items = split /\s+/, $line;

}

close IN;
close OUT;
