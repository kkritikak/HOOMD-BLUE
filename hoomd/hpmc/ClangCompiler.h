// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#pragma GCC diagnostic pop

#include <memory>
#include <sstream>
#include <string>
#include <vector>

/** Class that compiles C++ code strings to LLVM IR with clang.

    There are several one time LLVM initialization functions. This class uses the singleton pattern
    to call these only once.
*/
class ClangCompiler
    {
    public:
    /// delete the copy constructor
    ClangCompiler(ClangCompiler& other) = delete;

    /// delete the equals operator
    void operator=(const ClangCompiler&) = delete;

    /// Get an instance to the singleton class
    static std::shared_ptr<ClangCompiler> getClangCompiler();

    /// Compile the provided C++ code and return the LLVM module
    std::unique_ptr<llvm::Module> compileCode(const std::string& code,
                                              const std::vector<std::string>& user_args,
                                              llvm::LLVMContext& context,
                                              std::ostringstream& out);

    protected:
    ClangCompiler();

    static std::shared_ptr<ClangCompiler> m_clang_compiler;
    };
