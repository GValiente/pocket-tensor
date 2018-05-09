# catch-mini

catch-mini is a minimal subset of [Catch2 C++ test framework](https://github.com/catchorg/Catch2), implemented in less than 300 lines of code.  

It is distributed as a single header file. To replace Catch with catch-mini you only need to replace this header file, so if you only need Catch's basic functionality (`TEST_CASE` and `REQUIRE` macros), catch-mini can be useful to you.

## Why do we need yet another C++ test framework?

Good question. For C++ there are quite a number of established frameworks, including (but not limited to),
[Catch2](https://github.com/catchorg/Catch2),
[Google Test](http://code.google.com/p/googletest/),
[Boost.Test](http://www.boost.org/doc/libs/1_49_0/libs/test/doc/html/index.html),
[CppUnit](http://sourceforge.net/apps/mediawiki/cppunit/index.php?title=Main_Page),
[Cute](http://r2.ifs.hsr.ch/cute),
[many, many more](http://en.wikipedia.org/wiki/List_of_unit_testing_frameworks#C.2B.2B).

So what does catch-mini bring to the party that differentiates it from these?

* Quick and really easy to get started. Just download catch.hpp, `#include` it and you're away.
* No external dependencies. As long as you can compile C++11 and have a C++ standard library available.
* Write test cases as, self-registering, functions.
* Tests are named using free-form strings - no more couching names in legal identifiers.
* Fast build times, since it is implemented in less than 300 lines of code.

## Where to put catch-mini?

catch-mini is header only. All you need to do is drop the catch.hpp file somewhere reachable from your project - either in some central location you can set your header search path to find, or directly into your project tree itself!

The rest of this tutorial will assume that the catch-mini single-include header (or the include folder) is available unqualified - but you may need to prefix it with a folder name if necessary.

## Writing tests

Say you have written a function to calculate factorials and now you want to test it (let's leave aside TDD for now). 

```c++
unsigned int Factorial(unsigned int number) {
    return number <= 1 ? number : Factorial(number - 1) * number;
}
```

To keep things simple we'll put everything in a single file (<a href="#scaling-up">see later for more on how to structure your test files</a>).

```c++
#define CATCH_CONFIG_MAIN // This tells catch-mini to provide a main() - only do this in one cpp file
#include "catch.hpp"

unsigned int Factorial(unsigned int number) {
    return number <= 1 ? number : Factorial(number - 1) * number;
}

TEST_CASE("Factorials") {
    REQUIRE(Factorial(1) == 1);
    REQUIRE(Factorial(2) == 2);
    REQUIRE(Factorial(3) == 6);
    REQUIRE(Factorial(10) == 3628800);
}
```

This will compile to a complete executable which will execute all test cases (in this case there is just one), report any failures, report a summary of how many tests passed and failed and return the number of failed tests (useful for if you just want a yes/ no answer to: "did it work").

<a id="scaling-up"></a>
## Scaling up

To keep the tutorial simple we put all our code in a single file. This is fine to get started - and makes jumping into catch-mini even quicker and easier. As you write more real-world tests, though, this is not really the best approach.

The requirement is that the following block of code:

```c++
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
```

appears in _exactly one_ source file. Use as many additional cpp files (or whatever you call your implementation files) as you need for your tests, partitioned however makes most sense for your way of working. Each additional file need only ```#include "catch.hpp"``` - do not repeat the ```#define```!

Do not write your tests in header files!

