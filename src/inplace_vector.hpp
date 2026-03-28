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
public:
    using size_type              = std::size_t;
    using difference_type        = std::ptrdiff_t;
    using value_type             = std::remove_cv_t<T>;
    using pointer                = value_type*;
    using const_pointer          = const value_type*;
    using reference              = value_type&;
    using const_reference        = const value_type&;
    using iterator               = detail::inplace_vector::iterator<value_type>;
    using const_iterator         = detail::inplace_vector::iterator<const value_type>;
    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    constexpr inplace_vector() noexcept = default;

    constexpr explicit inplace_vector(size_type count)
    {
        capacity_check(count);
        exception_guard guard{this};
        while (count--) {
            unchecked_emplace_back();
        }
        guard.release();
    }

    constexpr inplace_vector(size_type count, const value_type& value)
    {
        capacity_check(count);
        exception_guard guard{this};
        while (count--) {
            unchecked_emplace_back(value);
        }
        guard.release();
    }

    template <std::input_iterator InputIt>
    constexpr inplace_vector(InputIt first, InputIt last)
    {
        if constexpr (std::random_access_iterator<InputIt>) {
            auto count = static_cast<size_type>(std::distance(first, last));
            capacity_check(count);
        }

        exception_guard guard{this};
        if constexpr (std::random_access_iterator<InputIt>) {
            for (; first != last; ++first) {
                unchecked_emplace_back(*first);
            }
        } else {
            for (; first != last; ++first) {
                emplace_back(*first);
            }
        }
        guard.release();
    }

    template <detail::container_compatible_range<T> R>
    constexpr inplace_vector(std::from_range_t, R&& rg)
        : inplace_vector(std::ranges::begin(rg), std::ranges::end(rg))
    {
    }

    constexpr inplace_vector(const inplace_vector&) noexcept
        requires std::is_trivially_copy_constructible_v<T> = default;
    constexpr inplace_vector(const inplace_vector& other)
    {
        exception_guard guard{this};
        for (auto first = other.begin(); first != other.end(); ++first) {
            unchecked_emplace_back(*first);
        }
        guard.release();
    }

    constexpr inplace_vector(inplace_vector&&) noexcept
        requires std::is_trivially_move_constructible_v<T> = default;
    constexpr inplace_vector(inplace_vector&& other)
        noexcept(N == 0 || std::is_nothrow_move_constructible_v<T>)
    {
        exception_guard guard{this};
        for (auto first = other.begin(); first != other.end(); ++first) {
            unchecked_emplace_back(std::move(*first));
        }
        guard.release();
    }

    constexpr inplace_vector(std::initializer_list<value_type> init)
        : inplace_vector(init.begin(), init.end())
    {
    }

    constexpr ~inplace_vector() requires std::is_trivially_destructible_v<T> = default;
    constexpr ~inplace_vector()
    {
        std::destroy_n(data(), size());
    }

    constexpr inplace_vector& operator=(const inplace_vector&) noexcept
        requires std::is_trivially_copy_assignable_v<T> = default;
    constexpr inplace_vector& operator=(const inplace_vector& other)
    {
        if (this != &other) {
            assign(other.begin(), other.end());
        }
        return *this;
    }

    constexpr inplace_vector& operator=(inplace_vector&&) noexcept
        requires std::is_trivially_move_assignable_v<T> = default;
    constexpr inplace_vector& operator=(inplace_vector&& other)
        noexcept(N == 0 || (std::is_nothrow_move_assignable_v<T> && std::is_nothrow_move_constructible_v<T>))
    {
        if (this != &other) {
            assign(std::make_move_iterator(other.begin()), std::make_move_iterator(other.end()));
        }
        return *this;
    }

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
            const auto last_copyable = first + std::min(size(), count);
            std::copy(first, last_copyable, data());
            for (auto pos = last_copyable; pos != last; ++pos) {
                unchecked_emplace_back(*pos);
            }
            truncate(count);
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

    constexpr iterator         end()      noexcept       { return iterator{data_end(), data(), size()}; }
    constexpr const_iterator   end()      const noexcept { return const_iterator{data_end(), data(), size()}; }
    constexpr const_iterator   cend()     const noexcept { return end(); }

    constexpr bool             empty()    const noexcept { return size() == 0; }
    constexpr size_type        size()     const noexcept { return storage_.size(); }
    static constexpr size_type max_size() noexcept       { return N; }
    static constexpr size_type capacity() noexcept       { return N; }

    void resize(size_type count)
    {
        if (size() > count) {
            truncate(count);
        }
        while (size() < count) {
            unchecked_emplace_back();
        }
    }

    void resize(size_type count, const value_type& value)
    {
        if (size() > count) {
            truncate(count);
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
        attic attic{this, pos, size() + 1};
        unchecked_emplace_back(value);
        attic.retrieve();
        return remove_const(pos);
    }

    constexpr iterator insert(const_iterator pos, value_type&& value)
    {
        capacity_check(size() + 1);
        attic attic{this, pos, size() + 1};
        unchecked_emplace_back(std::move(value));
        attic.retrieve();
        return remove_const(pos);
    }

    constexpr iterator insert(const_iterator pos, size_type count, const T& value)
    {
        capacity_check(size() + count);
        attic attic{this, pos, size() + count};
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
            attic attic{this, pos, size() + count};
            for (; count != 0; --count) {
                unchecked_emplace_back(*first++);
            }
            attic.retrieve();
            return remove_const(pos);
        } else {
            // We can't determine the size of the input range, so move the attic all the way up.
            attic attic{this, pos, capacity()};
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
        attic attic{this, pos, size() + 1};
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
        const auto pos = std::construct_at(data_end(), std::forward<Args>(args)...);
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
        std::destroy_at(data_end() - 1);
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
        truncate(0);
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
        truncate(new_size);
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
    class attic
    {
    public:
        /// Destructively move-construct elements in the range [save_pos..storage.size()) into the attic,
        /// [attic_end - storage.size() + save_pos..attic_end).
        /// @param vec The inplace_vetor in which to move elements.
        /// @param save_pos The position from which to move elements.
        /// @param attic_end The end position of the attic, into which to save elements.
        template <std::random_access_iterator Iterator>
        constexpr attic(inplace_vector* vec, Iterator save_pos, std::size_t attic_end)
            : vec_{vec}
            , begin_{attic_end}
            , end_{attic_end}
        {
            // Note that operator->() explicitly allows dereferencing at the end().
            const auto save_index = static_cast<std::size_t>(save_pos.operator->() - vec_->data());
        
            if (begin_ == vec_->size())
            {
                begin_ = save_index;
                vec_->size(begin_);
            } else {
                for (; vec_->size() != save_index; --begin_) {
                    const auto last_index = vec_->size() - 1;
                    std::construct_at(vec_->data() + (begin_ - 1), std::move(vec_->data()[last_index]));
                    std::destroy_at(vec_->data() + last_index);
                    vec_->size(last_index);
                }
            }
        }

        /// Destroy any remaining entries in the attic (typically only during an exception).
        constexpr ~attic()
        {
            std::destroy(vec_->data() + begin_, vec_->data() + end_);
        }

        /// Retrieve all elements from the attic, destructively move-constructing them if they are not already in their
        /// required location, and adjust the storage.size().
        constexpr void retrieve()
        {
            if (vec_->size() == begin_) {
                begin_ = end_;
                vec_->size(end_);
            } else {
                for (; begin_ != end_; ++begin_) {
                    vec_->unchecked_emplace_back(std::move(vec_->data()[begin_]));
                    std::destroy_at(vec_->data() + begin_);
                }
            }
        }

        /// Check that the position is not within the bounds of the attic or above, throwing bad_alloc if the check fails.
        /// @param pos The position to check.
        constexpr void capacity_check(std::size_t pos) const
        {
            if (pos >= begin_) {
                throw std::bad_alloc{};
            }
        }

    private:
        inplace_vector* vec_;
        size_type begin_;
        size_type end_;
    };
    friend class attic;

    class exception_guard
    {
    public:
        exception_guard(inplace_vector* vec) noexcept : vec_{vec} {}
        ~exception_guard()
        {
            if (vec_) {
                std::destroy_n(vec_->data(), vec_->size());
            }
        }
        void release() noexcept { vec_ = nullptr; }
    
    private:
        inplace_vector* vec_;
    };

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

    constexpr void          size(size_type n) noexcept { storage_.size(n); }
    constexpr pointer       data_end()        noexcept { return data() + size(); }
    constexpr const_pointer data_end()  const noexcept { return data() + size(); }

    constexpr iterator remove_const(const_iterator pos)
    {
        return begin() + (pos - begin());
    }

    constexpr void truncate(size_type n) noexcept
    {
        std::destroy_n(data() + n, size() - n);
        storage_.size(n);
    }

    [[no_unique_address]] detail::inplace_vector::storage<value_type, N> storage_;
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
