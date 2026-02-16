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

#include "detail/inplace_vector_forward.hpp"

#include <iterator>
#include <ranges>
#include <stdexcept>

namespace jell::detail::inplace_vector {

/// inplace_vector iterator.
/// Define CHECKED_ITERATORS to enable bounds checking.
template <typename T>
class iterator
{
private:
    friend iterator<std::add_const_t<T>>;

    template <typename U, std::size_t N>
        requires std::is_move_constructible_v<U> && std::is_move_assignable_v<U>
    friend class ::jell::inplace_vector;

    static constexpr inline bool is_const_iterator = std::is_const_v<T>;

    using iterator_traits    = std::iterator_traits<T*>;
    using non_const_iterator = iterator<std::remove_const_t<T>>;

public:
    using size_type          = std::size_t;
    using difference_type    = iterator_traits::difference_type;
    using value_type         = iterator_traits::value_type;
    using pointer            = iterator_traits::pointer;
    using reference          = iterator_traits::reference;
    using iterator_category  = iterator_traits::iterator_category;
    using iterator_concept   = iterator_traits::iterator_concept;

    constexpr iterator() noexcept = default;

    constexpr iterator(const iterator& other) noexcept = default;
    constexpr iterator(const non_const_iterator& other) noexcept requires is_const_iterator
        : pos_{other.pos_}
#if defined(CHECKED_ITERATORS)
        , first_{other.first_}
        , last_{other.last_}
#endif
    {}
    
    constexpr iterator& operator=(const iterator& other) noexcept = default;
    constexpr iterator& operator=(const non_const_iterator& other) noexcept requires is_const_iterator
    {
        pos_ = other.pos_;
#if defined(CHECKED_ITERATORS)
        first_ = other.first_;
        last_ = other.last_;
#endif
        return *this;
    }

    constexpr reference operator*() const
    {
        deref_check(pos_, "Dereferenced iterator (*) is out of range");
        return *pos_;
    }

    constexpr pointer operator->() const
    {
        // std::to_address() requires that operator-> be defined for the end-of-range.
        range_check(pos_, "Dereferenced iterator (->) is out of range");
        return pos_;
    }

    constexpr reference operator[](difference_type n) const
    {
        const auto pos{pos_ + n};
        range_check(pos, "Indexed iterator ([]) is out of range");
        return *pos;
    }
    
    constexpr iterator& operator++()
    {
        range_check(pos_ + 1, "Incremented iterator (++) is out of range");
        ++pos_;
        return *this;
    }

    constexpr iterator operator++(int)
    {
        auto tmp{*this};
        ++(*this);
        return tmp;
    }

    constexpr iterator& operator--()
    {
        range_check(pos_ - 1, "Decremented iterator (--) is out of range");
        --pos_;
        return *this;
    }

    constexpr iterator operator--(int)
    {
        auto tmp{*this};
        --(*this);
        return tmp;
    }

    constexpr iterator& operator+=(difference_type n)
    {
        const auto pos{pos_ + n};
        range_check(pos, "Computed iterator (+=/-=/+/-) is out of range");
        pos_ = pos;
        return *this;
    }

    constexpr iterator& operator-=(difference_type n)
    {
        *this += -n;
        return *this;
    }

    friend constexpr iterator operator+(const iterator& lhs, difference_type n)
    {
        auto tmp{lhs};
        tmp += n;
        return tmp;
    }

    friend constexpr iterator operator+(difference_type n, const iterator& rhs)
    {
        auto tmp{rhs};
        tmp += n;
        return tmp;
    }

    friend constexpr iterator operator-(const iterator& lhs, difference_type n)
    {
        auto tmp{lhs};
        tmp -= n;
        return tmp;
    }

    friend constexpr difference_type operator-(const iterator& lhs, const iterator& rhs)
    {
        return lhs.pos_ - rhs.pos_;
    }

    friend constexpr bool operator==(const iterator& lhs, const iterator& rhs) noexcept
    {
        return lhs.pos_ == rhs.pos_;
    }

    friend constexpr auto operator<=>(const iterator& lhs, const iterator& rhs) noexcept
    {
        return lhs.pos_ <=> rhs.pos_;
    }

private:
    constexpr void deref_check([[maybe_unused]] pointer pos, [[maybe_unused]] const char* message) const
    {
#if defined(CHECKED_ITERATORS)
        if (pos < first_ || pos >= last_) {
            throw std::range_error(message);
        }
#endif
    }

    constexpr void range_check([[maybe_unused]] pointer pos, [[maybe_unused]] const char* message) const
    {
#if defined(CHECKED_ITERATORS)
        if (pos < first_ || pos > last_) {
            throw std::range_error(message);
        }
#endif
    }

    constexpr iterator(pointer p, [[maybe_unused]] pointer base, [[maybe_unused]] size_type size) noexcept
        : pos_{p}
#if defined(CHECKED_ITERATORS)
        , first_{base}
        , last_{base + size}
#endif
    {}

    pointer pos_{nullptr};
#if defined(CHECKED_ITERATORS)
    pointer first_{nullptr};
    pointer last_{nullptr};
#endif
};

} // namespace jell::detail::inplace_vector
