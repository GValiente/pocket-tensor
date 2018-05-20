// Copyright (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
// claim that you wrote the original software. If you use this software
// in a product, an acknowledgment in the product documentation would be
// appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
// misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.

#ifndef CATCH_MINI_HPP
#define CATCH_MINI_HPP

// INTERNAL MACROS //

// Code to string conversion evaluation:
#define CATCH_MINI_STR(x) #x

// Unique names generation:
#define CATCH_MINI_UNIQUE_NAME(Name, Line) Name##Line

// Test cases:
#define CATCH_MINI_TESTCASE2(TestName, FunctionName, File, Line) \
    static void FunctionName(); \
    namespace { \
        bool CATCH_MINI_UNIQUE_NAME(autoReg, Line){ CatchMini::TestCase::create(&FunctionName, TestName, File, Line) }; \
    } \
    static void FunctionName()

#define CATCH_MINI_TESTCASE(TestName, File, Line) \
    CATCH_MINI_TESTCASE2(TestName, CATCH_MINI_UNIQUE_NAME(__CATCH__TEST__, Line), File, Line)

// Tests:
#define CATCH_MINI_TEST(Expr, File, Line) \
    do { \
        CatchMini::Assertion assertion(CATCH_MINI_STR(Expr), File, Line); \
        if(! (Expr)) { \
            throw assertion; \
        } \
    } while(false)


// PUBLIC MACROS //

#define TEST_CASE(TestName) CATCH_MINI_TESTCASE(TestName, __FILE__, __LINE__)

#define REQUIRE(...) CATCH_MINI_TEST((__VA_ARGS__), __FILE__, __LINE__)


// INTERNAL API //

namespace CatchMini
{

struct TestCase
{
    void(*function)();
    const char* name;
    const char* file;
    int line;

    static bool create(void(*function)(), const char* name, const char* file, int line);
};


struct Assertion
{
    const char* expr;
    const char* file;
    int line;

    Assertion(const char* _expr, const char* _file, int _line) noexcept;
};

}


// IMPLEMENTATION //

#ifdef CATCH_CONFIG_MAIN
#include <memory>
#include <vector>
#include <string>
#include <iostream>

namespace CatchMini
{

namespace
{
    struct StaticData
    {
        std::vector<std::pair<const char*, TestCase>> testCases;
        std::unique_ptr<TestCase> firstDuplicatedTestCase = nullptr;
        std::unique_ptr<TestCase> secondDuplicatedTestCase = nullptr;
    };

    StaticData* staticData;
    std::size_t numAssertions = 0;

    void initStaticData()
    {
        static StaticData* staticDataPtr = new StaticData();
        staticData = staticDataPtr;
    }

    void deleteStaticData() noexcept
    {
        delete staticData;
        staticData = nullptr;
    }

    void printTestCase(const TestCase& testCase)
    {
        std::cout << std::endl;
        std::cout << "-------------------------------------------------------------------------------" << std::endl;
        std::cout << testCase.name << std::endl;
        std::cout << "-------------------------------------------------------------------------------" << std::endl;
        std::cout << testCase.file << ':' << testCase.line << std::endl;
        std::cout << "..............................................................................." << std::endl;
        std::cout << std::endl;
    }

    void padStrings(std::string& a, std::string& b)
    {
        if(a.size() < b.size())
        {
            a = std::string(b.size() - a.size(), ' ') + a;
        }
        else if(b.size() < a.size())
        {
            b = std::string(a.size() - b.size(), ' ') + b;
        }
    }

    int launchTestCases()
    {
        int exitCode;
        initStaticData();

        const auto& firstDuplicatedTestCase = staticData->firstDuplicatedTestCase;

        if(firstDuplicatedTestCase)
        {
            const auto& secondDuplicatedTestCase = staticData->secondDuplicatedTestCase;
            std::cerr << "error: TEST_CASE( " << secondDuplicatedTestCase->name << " ) already defined." << std::endl;
            std::cerr << "\tFirst seen at " << firstDuplicatedTestCase->file << ':' << firstDuplicatedTestCase->line << std::endl;
            std::cerr << "\tRedefined at " << secondDuplicatedTestCase->file << ':' << secondDuplicatedTestCase->line << std::endl;
            exitCode = -1;
        }
        else
        {
            auto& testCases = staticData->testCases;
            std::size_t numFailedTestCases = 0;

            for(auto& testCasePair : testCases)
            {
                const TestCase& testCase = testCasePair.second;

                try
                {
                    testCase.function();
                }
                catch(const Assertion& assertion)
                {
                    printTestCase(testCase);
                    ++numFailedTestCases;

                    std::cout << assertion.file << ':' << assertion.line << ": FAILED:" << std::endl;
                    std::cout << "\tREQUIRE" << assertion.expr << std::endl;
                    std::cout << std::endl;
                }
                catch(const std::exception& exception)
                {
                    printTestCase(testCase);
                    ++numFailedTestCases;

                    std::cout << "Test case FAILED:" << std::endl;
                    std::cout << "due to unexpected exception with message:" << std::endl;
                    std::cout << "\t" << exception.what() << std::endl;
                    std::cout << std::endl;
                }
                catch(...)
                {
                    printTestCase(testCase);
                    ++numFailedTestCases;

                    std::cout << "Test case FAILED:" << std::endl;
                    std::cout << "due to unexpected exception with message:" << std::endl;
                    std::cout << "\tUnknown exception" << std::endl;
                    std::cout << std::endl;
                }
            }

            std::cout << "===============================================================================" << std::endl;

            if(numFailedTestCases)
            {
                auto numTestCasesStr = std::to_string(testCases.size());
                auto numAssertionsStr = std::to_string(numAssertions);
                padStrings(numTestCasesStr, numAssertionsStr);

                auto numPassedTestCasesStr = std::to_string(testCases.size() - numFailedTestCases);
                auto numPassedAssertionsStr = std::to_string(numAssertions - numFailedTestCases);
                padStrings(numPassedTestCasesStr, numPassedAssertionsStr);

                std::cout << "test cases: " << numTestCasesStr << " | " << numPassedTestCasesStr << " passed | " << numFailedTestCases << " failed" << std::endl;
                std::cout << "assertions: " << numAssertionsStr << " | " << numPassedAssertionsStr << " passed | " << numFailedTestCases << " failed" << std::endl;
                exitCode = 1;
            }
            else
            {
                std::cout << "All tests passed (" << numAssertions << " assertions in " << testCases.size() << " test cases)" << std::endl;
                exitCode = 0;
            }
        }

        deleteStaticData();
        return exitCode;
    }
}

bool TestCase::create(void(*function)(), const char* name, const char* file, int line)
{
    initStaticData();

    if(staticData->firstDuplicatedTestCase)
    {
        return false;
    }

    TestCase testCase{ function, name, file, line };
    auto& testCases = staticData->testCases;

    for(const auto& testCasePair : testCases)
    {
        if(testCasePair.first == name)
        {
            staticData->firstDuplicatedTestCase.reset(new TestCase(testCasePair.second));
            staticData->secondDuplicatedTestCase.reset(new TestCase(testCase));
            return false;
        }
    }

    testCases.push_back(std::make_pair(name, testCase));
    return true;
}

Assertion::Assertion(const char* _expr, const char* _file, int _line) noexcept :
    expr(_expr),
    file(_file),
    line(_line)
{
    ++numAssertions;
}

}

int main()
{
    std::cout << "This is a catch-mini host application" << std::endl << std::endl;
    return CatchMini::launchTestCases();
}
#endif

#endif
