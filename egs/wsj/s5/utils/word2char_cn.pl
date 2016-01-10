#!/usr/bin/env perl
# use warnings; #sed replacement for -w perl parameter

use 5.010;
use utf8;
use Encode;
use Fatal qw/open opendir/;

binmode(STDIN, ":encoding(utf8)");
binmode(STDOUT, ":encoding(utf8)");
binmode(STDERR, ":encoding(utf8)");

open IN, "<:encoding(utf8)", "$ARGV[0]";
open OUT, ">:encoding(utf8)", "$ARGV[1]";

while (my $line = <IN>) {
  chomp($line);
  @A = split /\s+/, $line;
  shift @A;
  foreach my $word (@A) {
    $word =~ s/(\p{Han})/$1 /g;
    $word = uc($word);
    print OUT " $word"
  }
  print OUT "\n";
}
