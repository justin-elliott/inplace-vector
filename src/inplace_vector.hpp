#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <ranges>
#include <type_traits>
#include <utility>

namespace jell {

namespace inplace_vector_detail {

template <typename R, typename T>
concept container_compatible_range =
    std::ranges::input_range<R> &&
    std::convertible_to<std::ranges::range_reference_t<R>, T>;

template <typename T, std::size_t N, typename IsTriviallyDestructible = std::is_trivially_destructible<T>>
struct storage
{
    [[nodiscard]] constexpr T* data() noexcept { return reinterpret_cast<T*>(data_); }
    [[nodiscard]] constexpr const T* data() const noexcept { return reinterpret_cast<const T*>(data_); }

    [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
    constexpr void set_size(size_t size) noexcept { size_ = size; }
    constexpr void clear() noexcept { size_ = 0; }

    alignas(T) std::byte data_[N * sizeof(T)];
    std::size_t size_{0};
};

template <typename T, std::size_t N>
struct storage<T, N, std::false_type> : storage<T, N, std::true_type>
{
    constexpr ~storage() { clear(); }

    constexpr void clear() noexcept
    {
        std::destroy_n(this->data(), this->size_);
        this->size_ = 0;
    }
};

template <typename T, typename IsTriviallyDestructible>
struct storage<T, 0, IsTriviallyDestructible>
{
    constexpr T* data() noexcept { return nullptr; }
    constexpr const T* data() const noexcept { return nullptr; }
    constexpr std::size_t size() const noexcept { return 0; }
    constexpr void set_size(size_t) noexcept {}
    constexpr void clear() noexcept {}
};

} // namespace inplace_vector_detail

template <typename T, std::size_t N>
    requires std::is_move_constructible_v<T> && std::is_move_assignable_v<T>
class inplace_vector
{
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    constexpr inplace_vector() noexcept = default;

    constexpr explicit inplace_vector(size_type count)
    {
        if (count > capacity()) {
            throw std::bad_alloc{};
        }
        exception_guard guard{this};
        while (count--) {
            emplace_back();
        }
        guard.release();
    }

    constexpr inplace_vector(size_type count, const value_type& value)
    {
        if (count > capacity()) {
            throw std::bad_alloc{};
        }
        exception_guard guard{this};
        while (count--) {
            emplace_back(value);
        }
        guard.release();
    }

    template <typename InputIt>
    constexpr inplace_vector(InputIt first, InputIt last)
    {
        exception_guard guard{this};
        for (; first != last; ++first) {
            emplace_back(*first);
        }
        guard.release();
    }

    template <inplace_vector_detail::container_compatible_range<T> R>
    constexpr inplace_vector(std::from_range_t, R&& rg)
        : inplace_vector(std::begin(rg), std::end(rg))
    {
    }

    constexpr inplace_vector(const inplace_vector& other)
        : inplace_vector(other.begin(), other.end())
    {
    }

    constexpr inplace_vector(inplace_vector&& other) noexcept(N == 0 || std::is_nothrow_move_constructible_v<T>)
    {
        exception_guard guard{this};
        for (iterator first = other.begin(), last = other.end(); first != last; ++first) {
            emplace_back(std::move(*first));
        }
        guard.release();
    }

    constexpr inplace_vector(std::initializer_list<value_type> init)
        : inplace_vector(init.begin(), init.end())
    {
    }

    constexpr ~inplace_vector() = default;

    constexpr reference operator[](size_type pos)
    {
        return *(data() + pos);
    }
    
    constexpr const_reference operator[](size_type pos) const
    {
        return *(data() + pos);
    }

    constexpr pointer data() noexcept { return storage_.data(); }
    constexpr const_pointer data() const noexcept { return storage_.data(); }
    
    constexpr iterator begin() noexcept { return iterator{data()}; }
    constexpr const_iterator begin() const noexcept { return const_iterator{data()}; }
    constexpr const_iterator cbegin() const noexcept { return begin(); }

    constexpr iterator end() noexcept { return iterator{data() + size()}; }
    constexpr const_iterator end() const noexcept { return const_iterator{data() + size()}; }
    constexpr const_iterator cend() const noexcept { return end(); }

    constexpr bool empty() const noexcept { return size() == 0; }
    constexpr size_type size() const noexcept { return storage_.size(); }
    constexpr size_type max_size() const noexcept { return N; }
    constexpr size_type capacity() const noexcept { return N; }

    template <typename... Args>
    constexpr reference emplace_back(Args&&... args)
    {
        if (size() == capacity()) {
            throw std::bad_alloc{};
        }
        pointer pos = data() + size();
        std::construct_at(pos, std::forward<Args>(args)...);
        storage_.set_size(size() + 1);
        return *pos;
    }

    constexpr void clear() noexcept { storage_.clear(); }

    constexpr friend bool operator==(const inplace_vector& lhs, const inplace_vector& rhs)
    {
        return std::equal(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
    }

    constexpr friend auto operator<=>(const inplace_vector& lhs, const inplace_vector& rhs)
    {
        return std::lexicographical_compare_three_way(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
    }

private:
    // In the event of an exception, free all vector elements.
    struct exception_guard
    {
        constexpr explicit exception_guard(inplace_vector* v) : v_{v} {}
        constexpr ~exception_guard()
        {
            if (v_) {
                v_->clear();
            }
        }
        constexpr void release() noexcept { v_ = nullptr; }

        inplace_vector* v_;
    };

    inplace_vector_detail::storage<T, N> storage_;
};

} // namespace jell
