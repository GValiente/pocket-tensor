/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_LOGGER_H
#define PT_LOGGER_H

#include <vector>
#include <iostream>

namespace pt
{

#define PT_LOG_ERROR std::cerr << '[' << pt::detail::baseName(__FILE__) << "::" << __LINE__ << "] "

template<typename T>
struct VectorPrinter
{
    const std::vector<T>& vector;

    friend std::ostream& operator<<(std::ostream& stream, const VectorPrinter& vectorPrinter)
    {
        stream << '[';

        for(std::size_t i = 0, l = vectorPrinter.vector.size(); i < l; ++i)
        {
            if(i)
            {
                stream << ", ";
            }

            stream << vectorPrinter.vector[i];
        }

        stream << ']';
        return stream;
    }
};

namespace detail
{
    template<unsigned int Size>
    constexpr const char* baseNameImpl(const char (&charArray)[Size], unsigned int index) noexcept
    {
        return index >= Size ?
                    charArray :
                    charArray[Size - index] == '/' ?
                        charArray + Size - index + 1 :
                        baseNameImpl(charArray, index + 1);
    }

    template<unsigned int Size>
    constexpr const char* baseName(const char (&charArray)[Size]) noexcept
    {
        return baseNameImpl(charArray, 2);
    }
}

}

#endif
