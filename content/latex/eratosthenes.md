Title: Programmatically-Generated LaTeX Sieve of Eratosthenes
Date: 2014-09-11 01:26
Tags: LaTeX, jokes, math

(tl;dr: see [PDF here](/pdf/sieve.pdf)). For a top-secret joke project I am learning to program in LaTeX as if it were a general-purpose language. LaTeX is my favourite way to write scientific papers, but no one should use it for programmatic logic. Along the way I have implemented a [Sieve of Eratosthenes](http://en.wikipedia.org/wiki/Sieve_of_Eratosthenes), a well-known algorithm for finding prime numbers. The natural typesetting ability of LaTeX allows me to easily generate an intuitive representation of the inner working of the algorithm.

<img style="float: center" src="/images/sieve_100.png" alt="Sieve of Eratosthenes">

Watch my rapid descent into TeX-induced insanity in my [Github repository](https://github.com/iank/latex-hacks/).

I first implemented [FizzBuzz](http://blog.codinghorror.com/why-cant-programmers-program/) using the LaTeX packages[^rawtex] "ifthen", "intcalc", and "forloop", which got me conditionals, simple arithmetic, and a useful control structure, respectively.

Then I found a brilliant "array" mechanism on the [TeX StackExchange](http://tex.stackexchange.com/questions/37426/create-an-array-of-variables), which I adapted to allow me to implement a [simple sieve (source)](https://github.com/iank/latex-hacks/blob/master/sieve/sieve_ugly.tex), the output of which is [an ugly numeric list (PDF)](/pdf/sieve_ugly.pdf).

But I thought I could have LaTeX represent the internal state of the algorithm as a colour-coded matrix at each point in its operation. I'm quite pleased with [the result (PDF)](/pdf/sieve.pdf), [(source)](https://github.com/iank/latex-hacks/blob/master/sieve/sieve.tex) which I think is an intuitive representation of how this simple algorithm works.

[^rawtex]: This can all be done in raw TeX, but I'm not a maniac.
