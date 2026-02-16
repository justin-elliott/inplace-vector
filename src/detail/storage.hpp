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
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using value_type      = T;
    using pointer         = T*;
    using const_pointer   = const T*;

    constexpr storage() noexcept = default;

    constexpr storage(const storage&) noexcept requires std::is_trivially_copy_constructible_v<T> = default;
    constexpr storage(const storage& other) noexcept(std::is_nothrow_copy_constructible_v<T>)
    {
        exception_guard([&] {
            for (; size_ != other.size_; ++size_) {
                construct_at(size_, other.data()[size_]);
            }
        });
    }

    constexpr storage(storage&&) noexcept requires std::is_trivially_move_constructible_v<T> = default;
    constexpr storage(storage&& other) noexcept(std::is_nothrow_move_constructible_v<T>)
    {
        exception_guard([&] {
            for (; size_ != other.size_; ++size_) {
                construct_at(size_, std::move(other.data()[size_]));
            }
        });
    }

    constexpr ~storage() requires std::is_trivially_destructible_v<T> = default;
    constexpr ~storage()
    {
        clear();
    }

    constexpr storage& operator=(const storage&) noexcept requires std::is_trivially_copy_assignable_v<T> = default;
    constexpr storage& operator=(const storage& other) noexcept(std::is_nothrow_copy_assignable_v<T>)
    {
        if (size_ > other.size_) {
            destroy(other.size_, size_);
            size_ = other.size_;
        }
        for (size_type i = 0; i != size_; ++i) {
            data()[i] = other.data()[i];
        }
        for (; size_ < other.size_; ++size_) {
            construct_at(size_, other.data()[size_]);
        }
        return *this;
    }

    constexpr storage& operator=(storage&&) noexcept requires std::is_trivially_move_assignable_v<T> = default;
    constexpr storage& operator=(storage&& other) noexcept(std::is_nothrow_move_assignable_v<T>)
    {
        if (size_ > other.size_) {
            destroy(other.size_, size_);
            size_ = other.size_;
        }
        for (size_type i = 0; i != size_; ++i) {
            data()[i] = std::move(other.data()[i]);
        }
        for (; size_ < other.size_; ++size_) {
            construct_at(size_, std::move(other.data()[size_]));
        }
        other.size_ = 0;
        return *this;
    }

    [[nodiscard]] constexpr pointer       data()       noexcept { return reinterpret_cast<T*>(data_); }
    [[nodiscard]] constexpr const_pointer data() const noexcept { return reinterpret_cast<const T*>(data_); }
    [[nodiscard]] constexpr size_type     size() const          { return size_; }
                  constexpr void          size(size_type n)     { size_ = n; }

    template <typename... Args>
    constexpr pointer construct_at(size_type i, Args&&... args)
    {
        return std::ranges::construct_at(data() + i, std::forward<Args>(args)...);
    }

    constexpr void destroy_at(size_type)   noexcept requires std::is_trivially_destructible_v<T> {}
    constexpr void destroy_at(size_type i) noexcept
    {
        std::ranges::destroy_at(data() + i);
    }
    
    constexpr void destroy(size_type, size_type) noexcept requires std::is_trivially_destructible_v<T> {}
    constexpr void destroy(size_type first, size_type last) noexcept
    {
        std::ranges::destroy(data() + first, data() + last);
    }

    constexpr void clear() noexcept
    {
        destroy(0, size_);
        size_ = 0;
    }

    template <typename Function, typename... Args>
    constexpr void exception_guard(Function&& function, Args&&... args)
    {
        try {
            std::invoke(std::forward<Function>(function), std::forward<Args>(args)...);
        } catch (...) {
            destroy(0, size_);
            throw;
        }
    }

private:
    alignas(value_type) std::byte data_[N * sizeof(value_type)];
    size_type size_{0};
};

/// Storage specialization for a zero-sized inplace_vector.
template <typename T>
struct storage<T, 0>
{
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using value_type      = T;
    using pointer         = T*;
    using const_pointer   = const T*;

    [[nodiscard]] constexpr pointer       data()       noexcept { return nullptr; }
    [[nodiscard]] constexpr const_pointer data() const noexcept { return nullptr; }
    [[nodiscard]] constexpr size_type     size() const          { return 0; }
                  constexpr void          size(size_type)       {}

    template <typename... Args>
    constexpr T* construct_at(size_type, Args&&...) noexcept { return nullptr; }

    constexpr void destroy_at(size_type) noexcept {}
    constexpr void destroy(size_type, size_type) noexcept {}
    constexpr void clear() noexcept {}

    template <typename Function, typename... Args>
    constexpr void exception_guard(Function&&, Args&&...) {}
};

} // namespace jell::detail::inplace_vector
