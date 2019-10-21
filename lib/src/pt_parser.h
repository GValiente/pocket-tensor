/*
 * pocket-tensor (c) 2019 Gustavo Valiente gustavo.valiente@protonmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef PT_PARSER_H
#define PT_PARSER_H

#include <istream>
#include "pt_logger.h"

namespace pt
{

namespace Parser
{
    template<typename T>
    bool parse(std::istream& stream, T& output)
    {
        stream.read(reinterpret_cast<char*>(&output), sizeof(T));

        auto readCharsCount = stream.gcount();

        if(readCharsCount != sizeof(T))
        {
            PT_LOG_ERROR << "Parse failed: " << readCharsCount << " - " << sizeof(T) << std::endl;
            return false;
        }

        return true;
    }

    template<typename T>
    bool parse(std::istream& stream, T* outputPtr, std::size_t outputCount) noexcept
    {
        if(! outputPtr)
        {
            PT_LOG_ERROR << "Output ptr is null" << std::endl;
            return false;
        }

        auto size = sizeof(T) * outputCount;
        stream.read(reinterpret_cast<char*>(outputPtr), size);

        auto readCharsCount = stream.gcount();

        if(readCharsCount != std::streamsize(size))
        {
            PT_LOG_ERROR << "Parse failed: " << readCharsCount << " - " << size << std::endl;
            return false;
        }

        return true;
    }
}

}

#endif
