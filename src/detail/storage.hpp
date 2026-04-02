// MIT License
//
// Copyright (c) 2026 Justin Elliott (github.com/justin-elliott)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "detail/uninitialized.hpp"

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace jell::detail::inplace_vector {

/// Storage for the inplace_vector.
/// @tparam T The element type.
/// @tparam N The number of elements to allocate in the storage.
template <typename T, std::size_t N>
class storage
{
public:
    [[nodiscard]] constexpr std::size_t size()        const noexcept { return size_; }
                  constexpr void        size(std::size_t n) noexcept { size_ = n; }
    [[nodiscard]] constexpr T*          data()              noexcept { return std::addressof(data_[0].value); }
    [[nodiscard]] constexpr const T*    data()        const noexcept { return std::addressof(data_[0].value); }

private:
    std::size_t size_{0};
    uninitialized<T> data_[N];
};

/// Storage specialization for a zero-sized inplace_vector.
template <typename T>
class storage<T, 0>
{
public:
    [[nodiscard]] constexpr std::size_t size()        const noexcept { return 0; }
                  constexpr void        size(std::size_t)   noexcept { }
    [[nodiscard]] constexpr T*          data()              noexcept { return nullptr; }
    [[nodiscard]] constexpr const T*    data()        const noexcept { return nullptr; }
};

} // namespace jell::detail::inplace_vector
