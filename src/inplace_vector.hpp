#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace jell {

template <typename T, std::size_t N>
    requires std::is_move_constructible_v<T> && std::is_move_assignable_v<T>
class inplace_vector;

namespace inplace_vector_detail {

template <typename R, typename T>
concept container_compatible_range =
    std::ranges::input_range<R> &&
    std::convertible_to<std::ranges::range_reference_t<R>, T>;

template <typename T, std::size_t N, typename IsTriviallyDestructible = std::is_trivially_destructible<T>>
struct storage
{
    alignas(T) std::byte data_[N * sizeof(T)];
    std::size_t size_{0};
};

template <typename T, std::size_t N>
struct storage<T, N, std::false_type>
{
    constexpr ~storage();

    alignas(T) std::byte data_[N * sizeof(T)];
    std::size_t size_{0};
};

template <typename T, typename IsTriviallyDestructible>
struct storage<T, 0, IsTriviallyDestructible>
{
};

template <typename T, std::size_t N>
concept non_empty = requires (storage<T, N> s) {
    s.data_;
    s.size_;
};

template <typename T, std::size_t N>
concept empty = !non_empty<T, N>;

template <typename T, std::size_t N>
[[nodiscard]] constexpr T* data(storage<T, N>& s) noexcept requires non_empty<T, N>
{
    return reinterpret_cast<T*>(s.data_);
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr const T* data(const storage<T, N>& s) noexcept requires non_empty<T, N>
{
    return reinterpret_cast<const T*>(s.data_);
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::size_t size(const storage<T, N>& s) noexcept requires non_empty<T, N>
{
    return s.size_;
}

template <typename T, std::size_t N>
constexpr void size(storage<T, N>& s, std::size_t size) noexcept requires non_empty<T, N>
{
    s.size_ = size;
}

template <typename T, std::size_t N>
constexpr void clear(storage<T, N>& s) noexcept
    requires non_empty<T, N> && std::is_trivially_destructible_v<T>
{
    s.size_ = 0;
}

template <typename T, std::size_t N>
constexpr void clear(storage<T, N>& s) noexcept
    requires non_empty<T, N> && (!std::is_trivially_destructible_v<T>)
{
    std::destroy_n(data(s), size(s));
    s.size_ = 0;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr T* data(storage<T, N>&) noexcept requires empty<T, N>
{
    return nullptr;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr const T* data(const storage<T, N>&) noexcept requires empty<T, N>
{
    return nullptr;
}

template <typename T, std::size_t N>
[[nodiscard]] constexpr std::size_t size(const storage<T, N>&) noexcept requires empty<T, N>
{
    return 0;
}

template <typename T, std::size_t N>
constexpr void size(storage<T, N>&, std::size_t) noexcept requires empty<T, N>
{
}

template <typename T, std::size_t N>
constexpr void clear(storage<T, N>&) noexcept requires empty<T, N>
{
}

template <typename T, std::size_t N>
constexpr storage<T, N, std::false_type>::~storage()
{
    clear(*this);
}

// define CHECKED_ITERATORS to enable bounds checking.
template <typename T>
class iterator
{
public:
    friend iterator<std::add_const_t<T>>;

    template <typename U, std::size_t N>
        requires std::is_move_constructible_v<U> && std::is_move_assignable_v<U>
    friend class ::jell::inplace_vector;

    using iterator_traits   = std::iterator_traits<T*>;
    using difference_type   = iterator_traits::difference_type;
    using value_type        = iterator_traits::value_type;
    using pointer           = iterator_traits::pointer;
    using reference         = iterator_traits::reference;
    using iterator_category = iterator_traits::iterator_category;
    using iterator_concept  = iterator_traits::iterator_concept;

    static constexpr inline bool is_const_iterator = std::is_const_v<T>;
    using non_const_iterator = iterator<std::remove_const_t<T>>;

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
        check_range_for_deref(pos_, "*iterator is out of range");
        return *pos_;
    }

    constexpr pointer operator->() const
    {
        // std::to_address() requires that operator-> be defined for the end-of-range.
        check_range_for_update(pos_, "iterator-> is out of range");
        return pos_;
    }

    constexpr reference operator[](difference_type n) const
    {
        const auto pos{pos_ + n};
        check_range_for_update(pos, "iterator[] is out of range");
        return *pos;
    }
    
    constexpr iterator& operator++()
    {
        check_range_for_update(pos_ + 1, "++iterator beyond the end of range");
        ++pos_;
        return *this;
    }

    constexpr iterator operator++(int)
    {
        iterator tmp{pos_};
        ++(*this);
        return tmp;
    }

    constexpr iterator& operator--()
    {
        check_range_for_update(pos_ - 1, "--iterator below the start of range");
        --pos_;
        return *this;
    }

    constexpr iterator operator--(int)
    {
        iterator tmp{pos_};
        --(*this);
        return tmp;
    }

    constexpr iterator& operator+=(difference_type n)
    {
        const auto pos{pos_ + n};
        check_range_for_update(pos, "iterator+= is out of range");
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
    constexpr void check_range_for_deref([[maybe_unused]] pointer pos, [[maybe_unused]] const char* message) const
    {
#if defined(CHECKED_ITERATORS)
        if (pos < first_ || pos >= last_) {
            throw std::range_error(message);
        }
#endif
    }

    constexpr void check_range_for_update([[maybe_unused]] pointer pos, [[maybe_unused]] const char* message) const
    {
#if defined(CHECKED_ITERATORS)
        if (pos < first_ || pos > last_) {
            throw std::range_error(message);
        }
#endif
    }

    constexpr iterator(pointer p, [[maybe_unused]] pointer base, [[maybe_unused]] std::size_t size) noexcept
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
    using iterator = inplace_vector_detail::iterator<T>;
    using const_iterator = inplace_vector_detail::iterator<const T>;
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
        other.clear();
        guard.release();
    }

    constexpr inplace_vector(std::initializer_list<value_type> init)
        : inplace_vector(init.begin(), init.end())
    {
    }

    constexpr ~inplace_vector() = default;

    constexpr inplace_vector& operator=(const inplace_vector& other) = default; // TODO

    constexpr inplace_vector& operator=(inplace_vector&& other)
        noexcept(N == 0 || (std::is_nothrow_move_assignable_v<T> && std::is_nothrow_move_constructible_v<T>));

    constexpr reference operator[](size_type pos)
    {
        return *(data() + pos);
    }
    
    constexpr const_reference operator[](size_type pos) const
    {
        return *(data() + pos);
    }

    constexpr pointer data() noexcept { return inplace_vector_detail::data(storage_); }
    constexpr const_pointer data() const noexcept { return inplace_vector_detail::data(storage_); }
    
    constexpr iterator begin() noexcept { return iterator{data(), data(), size()}; }
    constexpr const_iterator begin() const noexcept { return const_iterator{data(), data(), size()}; }
    constexpr const_iterator cbegin() const noexcept { return begin(); }

    constexpr iterator end() noexcept { return iterator{data() + size(), data(), size()}; }
    constexpr const_iterator end() const noexcept { return const_iterator{data() + size(), data(), size()}; }
    constexpr const_iterator cend() const noexcept { return end(); }

    constexpr bool empty() const noexcept { return size() == 0; }
    constexpr size_type size() const noexcept { return inplace_vector_detail::size(storage_); }
    static constexpr size_type max_size() noexcept { return N; }
    static constexpr size_type capacity() noexcept { return N; }

    template <typename... Args>
    constexpr reference emplace_back(Args&&... args)
    {
        if (size() == capacity()) {
            throw std::bad_alloc{};
        }
        pointer pos = data() + size();
        std::construct_at(pos, std::forward<Args>(args)...);
        inplace_vector_detail::size(storage_, size() + 1);
        return *pos;
    }

    constexpr void clear() noexcept { inplace_vector_detail::clear(storage_); }

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
