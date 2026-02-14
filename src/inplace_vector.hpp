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

template <typename T>
struct exception_guard
{
    constexpr explicit exception_guard(T* guarded) : guarded_{guarded} {}
    constexpr ~exception_guard()
    {
        if (guarded_) {
            guarded_->clear();
        }
    }
    constexpr void release() noexcept { guarded_ = nullptr; }

    T* guarded_;
};

template <typename T, std::size_t N>
struct storage
{
    constexpr storage() noexcept = default;

    constexpr storage(const storage&) noexcept requires std::is_trivially_copy_constructible_v<T> = default;
    constexpr storage(const storage& other) noexcept(std::is_nothrow_copy_constructible_v<T>)
    {
        exception_guard guard{this};
        for (; size_ != other.size_; ++size_) {
            construct_at(size_, other.data()[size_]);
        }
        guard.release();
    }

    constexpr storage(storage&&) noexcept requires std::is_trivially_move_constructible_v<T> = default;
    constexpr storage(storage&& other) noexcept(std::is_nothrow_move_constructible_v<T>)
    {
        exception_guard guard{this};
        for (; size_ != other.size_; ++size_) {
            construct_at(size_, std::move(other.data()[size_]));
        }
        other.size_ = 0;
        guard.release();
    }

    constexpr ~storage() requires std::is_trivially_destructible_v<T> = default;
    constexpr ~storage()
    {
        clear();
    }

    constexpr storage& operator=(const storage&) noexcept requires std::is_trivially_copy_assignable_v<T> = default;
    constexpr storage& operator=(const storage& other) noexcept(std::is_nothrow_copy_assignable_v<T>)
    {
        for (; size_ > other.size_; --size_) {
            destroy_at(size_ - 1);
        }
        for (std::size_t i = 0; i != size_; ++i) {
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
        for (; size_ > other.size_; --size_) {
            destroy_at(size_ - 1);
        }
        for (std::size_t i = 0; i != size_; ++i) {
            data()[i] = std::move(other.data()[i]);
        }
        for (; size_ < other.size_; ++size_) {
            construct_at(size_, std::move(other.data()[size_]));
        }
        other.size_ = 0;
        return *this;
    }

    [[nodiscard]] constexpr T*          data()       noexcept { return reinterpret_cast<T*>(data_); }
    [[nodiscard]] constexpr const T*    data() const noexcept { return reinterpret_cast<const T*>(data_); }
    [[nodiscard]] constexpr std::size_t size() const          { return size_; }
                  constexpr void        size(std::size_t n)   { size_ = n; }

    template <typename... Args>
    constexpr T* construct_at(std::size_t i, Args&&... args)
    {
        return std::ranges::construct_at(data() + i, std::forward<Args>(args)...);
    }

    constexpr void destroy_at(std::size_t)   noexcept requires std::is_trivially_destructible_v<T> {}
    constexpr void destroy_at(std::size_t i) noexcept
    {
        std::ranges::destroy_at(data() + i);
    }
    
    constexpr void destroy(std::size_t, std::size_t) noexcept requires std::is_trivially_destructible_v<T> {}
    constexpr void destroy(std::size_t first, std::size_t last) noexcept
    {
        std::ranges::destroy(data() + first, data() + last);
    }

    constexpr void clear() noexcept
    {
        destroy(0, size_);
        size_ = 0;
    }

    alignas(T) std::byte data_[N * sizeof(T)];
    std::size_t size_{0};
};

template <typename T>
struct storage<T, 0>
{
    [[nodiscard]] constexpr T*          data()       noexcept { return nullptr; }
    [[nodiscard]] constexpr const T*    data() const noexcept { return nullptr; }
    [[nodiscard]] constexpr std::size_t size() const          { return 0; }
                  constexpr void        size(std::size_t)     {}

    template <typename... Args>
    constexpr T* construct_at(std::size_t, Args&&...) noexcept { return nullptr; }

    constexpr void destroy_at(std::size_t) noexcept {}
    constexpr void destroy(std::size_t, std::size_t) noexcept {}
    constexpr void clear() noexcept {}
};

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
        inplace_vector_detail::exception_guard guard{this};
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
        inplace_vector_detail::exception_guard guard{this};
        while (count--) {
            emplace_back(value);
        }
        guard.release();
    }

    template <typename InputIt>
    constexpr inplace_vector(InputIt first, InputIt last)
    {
        inplace_vector_detail::exception_guard guard{this};
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
        resize(size() + count, value);
    }

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
    
    constexpr iterator begin() noexcept { return iterator{data(), data(), size()}; }
    constexpr const_iterator begin() const noexcept { return const_iterator{data(), data(), size()}; }
    constexpr const_iterator cbegin() const noexcept { return begin(); }

    constexpr iterator end() noexcept { return iterator{data() + size(), data(), size()}; }
    constexpr const_iterator end() const noexcept { return const_iterator{data() + size(), data(), size()}; }
    constexpr const_iterator cend() const noexcept { return end(); }

    constexpr bool empty() const noexcept { return size() == 0; }
    constexpr size_type size() const noexcept { return storage_.size(); }
    static constexpr size_type max_size() noexcept { return N; }
    static constexpr size_type capacity() noexcept { return N; }

    void resize(size_type count)
    {
        while (size() > count) {
            pop_back();
        }
        while (size() < count) {
            emplace_back();
        }
    }

    void resize(size_type count, const value_type& value)
    {
        while (size() > count) {
            pop_back();
        }
        while (size() < count) {
            emplace_back(value);
        }
    }

    template <typename... Args>
    constexpr reference emplace_back(Args&&... args)
    {
        if (size() == capacity()) {
            throw std::bad_alloc{};
        }
        const auto pos = storage_.construct_at(storage_.size(), std::forward<Args>(args)...);
        storage_.size(storage_.size() + 1);
        return *pos;
    }

    constexpr void pop_back()
    {
        storage_.destroy_at(size() - 1);
        storage_.size(size() - 1);
    }

    constexpr void clear() noexcept
    {
        storage_.clear();
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
    [[no_unique_address]] inplace_vector_detail::storage<T, N> storage_;
};

} // namespace jell
