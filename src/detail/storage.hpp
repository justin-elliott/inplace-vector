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
    [[nodiscard]] constexpr T*          data()              noexcept { return data_.data; }
    [[nodiscard]] constexpr const T*    data()        const noexcept { return data_.data; }

private:
    union uninit
    {
        constexpr uninit() noexcept requires std::is_trivially_constructible_v<T> = default;
        constexpr uninit() noexcept {}

        constexpr uninit(const uninit&) noexcept requires std::is_trivially_copy_constructible_v<T> = default;
        constexpr uninit(const uninit&) noexcept {}

        constexpr uninit(uninit&&) noexcept requires std::is_trivially_move_constructible_v<T> = default;
        constexpr uninit(uninit&&) noexcept {}

        constexpr ~uninit() requires std::is_trivially_destructible_v<T> = default;
        constexpr ~uninit() {}

        constexpr uninit& operator=(const uninit&) noexcept requires std::is_trivially_copy_assignable_v<T> = default;
        constexpr uninit& operator=(const uninit&) noexcept { return *this; }

        constexpr uninit& operator=(uninit&&) noexcept requires std::is_trivially_move_assignable_v<T> = default;
        constexpr uninit& operator=(uninit&&) noexcept { return *this; }

        T data[N];
    };

    std::size_t size_{0};
    uninit data_;
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
