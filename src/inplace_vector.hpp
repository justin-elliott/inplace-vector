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

#include "detail/attic.hpp"
#include "detail/container_compatible_range.hpp"
#include "detail/iterator.hpp"
#include "detail/storage.hpp"

#include <algorithm>
#include <format>

namespace jell {

/// A dynamically-resizable array with contiguous inplace storage.
/// @tparam T The element type.
/// @tparam N The maximum number of elements that can be stored in the container.
template <typename T, std::size_t N>
    requires std::is_move_constructible_v<T> && std::is_move_assignable_v<T>
class inplace_vector
{
private:
    using storage_type           = detail::inplace_vector::storage<T, N>;
    using attic_type             = detail::inplace_vector::attic<T, N>;

public:
    using size_type              = storage_type::size_type;
    using difference_type        = storage_type::difference_type;
    using value_type             = storage_type::value_type;
    using pointer                = storage_type::pointer;
    using const_pointer          = storage_type::const_pointer;
    using reference              = value_type&;
    using const_reference        = const value_type&;
    using iterator               = detail::inplace_vector::iterator<T>;
    using const_iterator         = detail::inplace_vector::iterator<const T>;
    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    constexpr inplace_vector() noexcept = default;

    constexpr explicit inplace_vector(size_type count)
    {
        capacity_check(count);
        storage_.exception_guard([&] {
            while (count--) {
                unchecked_emplace_back();
            }
        });
    }

    constexpr inplace_vector(size_type count, const value_type& value)
    {
        capacity_check(count);
        storage_.exception_guard([&] {
            while (count--) {
                unchecked_emplace_back(value);
            }
        });
    }

    template <std::input_iterator InputIt>
    constexpr inplace_vector(InputIt first, InputIt last)
    {
        storage_.exception_guard([&] {
            assign(std::move(first), std::move(last));
        });
    }

    template <detail::container_compatible_range<T> R>
    constexpr inplace_vector(std::from_range_t, R&& rg)
        : inplace_vector(std::ranges::begin(rg), std::ranges::end(rg))
    {
    }

    constexpr inplace_vector(const inplace_vector& other) = default;
    constexpr inplace_vector(inplace_vector&& other)
        noexcept(N == 0 || std::is_nothrow_move_constructible_v<T>) = default;

    constexpr inplace_vector(std::initializer_list<value_type> init)
        : inplace_vector(init.begin(), init.end())
    {
    }

    constexpr ~inplace_vector() = default;

    constexpr inplace_vector& operator=(const inplace_vector& other) = default;
    constexpr inplace_vector& operator=(inplace_vector&& other)
        noexcept(N == 0 || (std::is_nothrow_move_assignable_v<T> && std::is_nothrow_move_constructible_v<T>)) = default;

    constexpr void assign(size_type count, const value_type& value)
    {
        capacity_check(count);
        const auto last = begin() + std::min(size(), count);
        for (auto first = begin(); first != last; ++first) {
            *first = value;
        }
        resize(count, value);
    }

    template <std::input_iterator InputIt>
    constexpr void assign(InputIt first, InputIt last)
    {
        if constexpr (std::random_access_iterator<InputIt>) {
            auto count = static_cast<size_type>(std::distance(first, last));
            capacity_check(count);
            const auto dest_end = begin() + std::min(size(), count);
            for (auto dest = begin(); dest != dest_end; ++dest, ++first) {
                *dest = *first;
            }
            for (; first != last; ++first) {
                unchecked_emplace_back(*first);
            }
            resize(count);
        } else {
            clear();
            for (; first != last; ++first) {
                emplace_back(*first);
            }
        }
    }

    constexpr void assign(std::initializer_list<value_type> init)
    {
        assign(init.begin(), init.end());
    }

    template <detail::container_compatible_range<T> R>
    constexpr void assign_range(R&& rg)
    {
        assign(std::ranges::begin(rg), std::ranges::end(rg));
    }

    constexpr reference at(size_type pos)
    {
        range_check(pos);
        return data()[pos];
    }

    constexpr const_reference at(size_type pos) const
    {
        range_check(pos);
        return data()[pos];
    }

    constexpr reference        operator[](size_type pos)       { return data()[pos]; }
    constexpr const_reference  operator[](size_type pos) const { return data()[pos]; }

    constexpr reference        front()                   { return data()[0]; }
    constexpr const_reference  front()    const          { return data()[0]; }

    constexpr reference        back()                    { return data()[size() - 1]; }
    constexpr const_reference  back()     const          { return data()[size() - 1]; }

    constexpr pointer          data()     noexcept       { return storage_.data(); }
    constexpr const_pointer    data()     const noexcept { return storage_.data(); }
    
    constexpr iterator         begin()    noexcept       { return iterator{data(), data(), size()}; }
    constexpr const_iterator   begin()    const noexcept { return const_iterator{data(), data(), size()}; }
    constexpr const_iterator   cbegin()   const noexcept { return begin(); }

    constexpr iterator         end()      noexcept       { return iterator{data() + size(), data(), size()}; }
    constexpr const_iterator   end()      const noexcept { return const_iterator{data() + size(), data(), size()}; }
    constexpr const_iterator   cend()     const noexcept { return end(); }

    constexpr bool             empty()    const noexcept { return size() == 0; }
    constexpr size_type        size()     const noexcept { return storage_.size(); }
    static constexpr size_type max_size() noexcept       { return N; }
    static constexpr size_type capacity() noexcept       { return N; }

    void resize(size_type count)
    {
        if (size() > count) {
            storage_.destroy(count, size());
            storage_.size(count);
        }
        while (size() < count) {
            unchecked_emplace_back();
        }
    }

    void resize(size_type count, const value_type& value)
    {
        if (size() > count) {
            storage_.destroy(count, size());
            storage_.size(count);
        }
        while (size() < count) {
            unchecked_emplace_back(value);
        }
    }

    static constexpr void reserve(size_type new_capacity)
    {
        capacity_check(new_capacity);
    }

    static constexpr void shrink_to_fit() noexcept {}

    constexpr iterator insert(const_iterator pos, const value_type& value)
    {
        capacity_check(size() + 1);
        attic_type attic{storage_, pos, size() + 1};
        unchecked_emplace_back(value);
        attic.retrieve();
        return remove_const(pos);
    }

    constexpr iterator insert(const_iterator pos, value_type&& value)
    {
        capacity_check(size() + 1);
        attic_type attic{storage_, pos, size() + 1};
        unchecked_emplace_back(std::move(value));
        attic.retrieve();
        return remove_const(pos);
    }

    constexpr iterator insert(const_iterator pos, size_type count, const T& value)
    {
        capacity_check(size() + count);
        attic_type attic{storage_, pos, size() + count};
        for (; count != 0; --count) {
            unchecked_emplace_back(value);
        }
        attic.retrieve();
        return remove_const(pos);
    }

    template <std::input_iterator InputIt>
    constexpr iterator insert(const_iterator pos, InputIt first, InputIt last)
    {
        if constexpr (std::random_access_iterator<InputIt>) {
            // We can determine the size of the input range.
            auto count = static_cast<size_type>(std::distance(first, last));
            capacity_check(size() + count);
            attic_type attic{storage_, pos, size() + count};
            for (; count != 0; --count) {
                unchecked_emplace_back(*first++);
            }
            attic.retrieve();
            return remove_const(pos);
        } else {
            // We can't determine the size of the input range, so move the attic all the way up.
            attic_type attic{storage_, pos, capacity()};
            while (first != last) {
                attic.capacity_check(size());
                unchecked_emplace_back(*first++);
            }
            attic.retrieve(); // Moves the attic elements back into place.
            return remove_const(pos);
        }
    }

    constexpr iterator insert(const_iterator pos, std::initializer_list<T> init)
    {
        return insert(pos, init.begin(), init.end());
    }

    template <detail::container_compatible_range<T> R>
    constexpr iterator insert_range(const_iterator pos, R&& rg)
    {
        return insert(pos, std::ranges::begin(rg), std::ranges::end(rg));
    }

    template <typename... Args>
    constexpr iterator emplace(const_iterator pos, Args&&... args)
    {
        capacity_check(size() + 1);
        attic_type attic{storage_, pos, size() + 1};
        unchecked_emplace_back(std::forward<Args>(args)...);
        attic.retrieve();
        return remove_const(pos);
    }

    template <typename... Args>
    constexpr reference emplace_back(Args&&... args)
    {
        capacity_check(size() + 1);
        return unchecked_emplace_back(std::forward<Args>(args)...);
    }

    template <typename... Args>
    constexpr pointer try_emplace_back(Args&&... args)
    {
        if (size() >= capacity()) {
            return nullptr;
        }
        return std::addressof(unchecked_emplace_back(std::forward<Args>(args)...));
    }

    template <typename... Args>
    constexpr reference unchecked_emplace_back(Args&&... args)
    {
        const auto pos = storage_.construct_at(storage_.size(), std::forward<Args>(args)...);
        storage_.size(storage_.size() + 1);
        return *pos;
    }

    constexpr reference push_back(const value_type& value)
    {
        return emplace_back(value);
    }

    constexpr reference push_back(value_type&& value)
    {
        return emplace_back(std::move(value));
    }

    constexpr pointer try_push_back(const value_type& value)
    {
        return try_emplace_back(value);
    }

    constexpr pointer try_push_back(value_type&& value)
    {
        return try_emplace_back(std::move(value));
    }

    constexpr reference unchecked_push_back(const value_type& value)
    {
        return unchecked_emplace_back(value);
    }

    constexpr reference unchecked_push_back(value_type&& value)
    {
        return unchecked_emplace_back(std::move(value));
    }

    constexpr void pop_back()
    {
        storage_.destroy_at(size() - 1);
        storage_.size(size() - 1);
    }

    template <detail::container_compatible_range<T> R>
    constexpr void append_range(R&& rg)
    {
        capacity_check(size() + std::ranges::size(rg));
        for (auto&& value : rg) {
            unchecked_emplace_back(std::forward<decltype(value)>(value));
        }
    }

    template <detail::container_compatible_range<T> R>
    constexpr std::ranges::borrowed_iterator_t<R> try_append_range(R&& rg)
    {
        const auto available = capacity() - size();
        auto count = std::min(std::ranges::size(rg), available);
        auto pos = std::ranges::begin(rg);
        for (; count != 0; ++pos, --count) {
            unchecked_emplace_back(*pos);
        }
        return pos;
    }

    constexpr void clear() noexcept
    {
        storage_.clear();
    }

    constexpr iterator erase(const_iterator pos)
    {
        return erase(pos, pos + 1);
    }

    constexpr iterator erase(const_iterator first, const_iterator last)
    {
        auto dst = remove_const(first);
        auto src = remove_const(last);
        while (src != end()) {
            *dst++ = std::move(*src++);
        }
        const auto new_size = dst - begin();
        storage_.destroy(new_size, size());
        storage_.size(new_size);
        return remove_const(first);
    }

    constexpr void swap(inplace_vector& other)
        noexcept(N == 0 || (std::is_nothrow_swappable_v<T> && std::is_nothrow_move_constructible_v<T>))
    {
        auto swap_count = std::min(size(), other.size());
        size_type i = 0;
        for (; i < swap_count; ++i) {
            std::swap((*this)[i], other[i]);
        }
        if (i < other.size()) {
            const auto first = other.begin() + i;
            append_range(std::ranges::subrange(first, other.end()) | std::views::as_rvalue);
            other.erase(first, other.end());
        } else if (i < size()) {
            const auto first = begin() + i;
            other.append_range(std::ranges::subrange(first, end()) | std::views::as_rvalue);
            erase(first, end());
        }
    }

    constexpr friend bool operator==(const inplace_vector& lhs, const inplace_vector& rhs)
    {
        return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
    }

    constexpr friend auto operator<=>(const inplace_vector& lhs, const inplace_vector& rhs)
    {
        return std::lexicographical_compare_three_way(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
    }

private:
    constexpr void range_check(size_type pos) const
    {
        if (pos >= size())
        {
            throw std::out_of_range{std::format("pos >= size() [{} >= {}]", pos, size())};
        }
    }

    static constexpr void capacity_check(size_type size)
    {
        if (size > capacity())
        {
            throw std::bad_alloc{};
        }
    }

    constexpr iterator remove_const(const_iterator pos)
    {
        return begin() + (pos - begin());
    }

    [[no_unique_address]] storage_type storage_;
};

} // namespace jell

namespace std {

template <typename T, std::size_t N>
constexpr void swap(jell::inplace_vector<T, N>& lhs, jell::inplace_vector<T, N>& rhs)
    noexcept(N == 0 || (std::is_nothrow_swappable_v<T> && std::is_nothrow_move_constructible_v<T>))
{
    lhs.swap(rhs);
}

template <typename T, std::size_t N, typename U = T>
constexpr auto erase(jell::inplace_vector<T, N>& c, const U& value)
{
    using vector = jell::inplace_vector<T, N>;
    auto iter = std::remove(c.begin(), c.end(), value);
    auto erase_count = static_cast<typename vector::size_type>(std::distance(iter, c.end()));
    c.erase(iter, c.end());
    return erase_count;
}

template <typename T, std::size_t N, typename Predicate>
constexpr auto erase_if(jell::inplace_vector<T, N>& c, Predicate predicate)
{
    using vector = jell::inplace_vector<T, N>;
    auto iter = std::remove_if(c.begin(), c.end(), predicate);
    auto erase_count = static_cast<typename vector::size_type>(std::distance(iter, c.end()));
    c.erase(iter, c.end());
    return erase_count;
}

} // namespace std
