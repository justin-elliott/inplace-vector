#pragma once

#include <algorithm>
#include <cstddef>
#include <format>
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
class exception_guard
{
public:
    constexpr explicit exception_guard(T* guarded) : guarded_{guarded} {}
    constexpr ~exception_guard()
    {
        if (guarded_) {
            guarded_->clear();
        }
    }
    constexpr void release() noexcept { guarded_ = nullptr; }

private:
    T* guarded_;
};

template <typename T, std::size_t N>
class storage
{
public:
    using size_type = std::size_t;

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

    [[nodiscard]] constexpr T*          data()       noexcept { return reinterpret_cast<T*>(data_); }
    [[nodiscard]] constexpr const T*    data() const noexcept { return reinterpret_cast<const T*>(data_); }
    [[nodiscard]] constexpr size_type   size() const          { return size_; }
                  constexpr void        size(size_type n)     { size_ = n; }

    template <typename... Args>
    constexpr T* construct_at(size_type i, Args&&... args)
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

private:
    alignas(T) std::byte data_[N * sizeof(T)];
    size_type size_{0};
};

template <typename T>
struct storage<T, 0>
{
    using size_type = std::size_t;

    [[nodiscard]] constexpr T*          data()       noexcept { return nullptr; }
    [[nodiscard]] constexpr const T*    data() const noexcept { return nullptr; }
    [[nodiscard]] constexpr size_type   size() const          { return 0; }
                  constexpr void        size(size_type)       {}

    template <typename... Args>
    constexpr T* construct_at(size_type, Args&&...) noexcept { return nullptr; }

    constexpr void destroy_at(size_type) noexcept {}
    constexpr void destroy(size_type, size_type) noexcept {}
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
    using size_type         = std::size_t;
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
        iterator tmp{pos_};
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
        iterator tmp{pos_};
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
        capacity_check(count);
        inplace_vector_detail::exception_guard guard{this};
        while (count--) {
            emplace_back();
        }
        guard.release();
    }

    constexpr inplace_vector(size_type count, const value_type& value)
    {
        capacity_check(count);
        inplace_vector_detail::exception_guard guard{this};
        while (count--) {
            emplace_back(value);
        }
        guard.release();
    }

    template <std::input_iterator InputIt>
    constexpr inplace_vector(InputIt first, InputIt last)
    {
        inplace_vector_detail::exception_guard guard{this};
        assign(std::move(first), std::move(last));
        guard.release();
    }

    template <inplace_vector_detail::container_compatible_range<T> R>
    constexpr inplace_vector(std::from_range_t, R&& rg)
    {
        inplace_vector_detail::exception_guard guard{this};
        for (auto&& value : rg) {
            emplace_back(std::forward<decltype(value)>(value));
        }
        guard.release();
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

    template <std::input_iterator InputIt>
    constexpr void assign(InputIt first, InputIt last)
    {
        for (; first != last; ++first) {
            emplace_back(*first);
        }
    }

    constexpr void assign(std::initializer_list<value_type> init)
    {
        assign(init.begin(), init.end());
    }

    template <inplace_vector_detail::container_compatible_range<T> R>
    constexpr void assign_range(R&& rg)
    {
        for (auto&& value : rg) {
            emplace_back(std::forward<decltype(value)>(value));
        }
    }

    constexpr reference at(size_type pos)
    {
        range_check(pos);
        return *(data() + pos);
    }

    constexpr const_reference at(size_type pos) const
    {
        range_check(pos);
        return *(data() + pos);
    }

    constexpr reference        operator[](size_type pos)       { return *(data() + pos); }
    constexpr const_reference  operator[](size_type pos) const { return *(data() + pos); }

    constexpr reference        front()                   { return *data(); }
    constexpr const_reference  front()    const          { return *data(); }

    constexpr reference        back()                    { return *(data() + size() - 1); }
    constexpr const_reference  back()     const          { return *(data() + size() - 1); }

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
            emplace_back();
        }
    }

    void resize(size_type count, const value_type& value)
    {
        if (size() > count) {
            storage_.destroy(count, size());
            storage_.size(count);
        }
        while (size() < count) {
            emplace_back(value);
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
        attic attic{*this, size() + 1, pos};
        emplace_back(value);
        attic.retrieve();
        return remove_const(pos);
    }

    constexpr iterator insert(const_iterator pos, value_type&& value)
    {
        capacity_check(size() + 1);
        attic attic{*this, size() + 1, pos};
        emplace_back(std::move(value));
        attic.retrieve();
        return remove_const(pos);
    }

    constexpr iterator insert(const_iterator pos, size_type count, const T& value)
    {
        capacity_check(size() + count);
        attic attic{*this, size() + count, pos};
        for (; count != 0; --count) {
            emplace_back(value);
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
            attic attic{*this, size() + count, pos};
            for (; count != 0; --count) {
                emplace_back(*first++);
            }
            attic.retrieve();
            return remove_const(pos);
        } else {
            // We can't determine the size of the input range, so move the attic all the way up.
            attic attic{*this, capacity(), pos};
            while (first != last) {
                attic.capacity_check(size());
                emplace_back(*first++);
            }
            attic.retrieve(); // Moves the attic elements back into place.
            return remove_const(pos);
        }
    }

    constexpr iterator insert(const_iterator pos, std::initializer_list<T> init)
    {
        return insert(pos, init.begin(), init.end());
    }

    template <inplace_vector_detail::container_compatible_range<T> R>
    constexpr iterator insert_range(const_iterator pos, R&& rg)
    {
        auto count = static_cast<size_type>(std::ranges::size(rg));
        capacity_check(size() + count);
        attic attic{*this, size() + count, pos};
        for (auto&& value : rg) {
            emplace_back(std::forward<decltype(value)>(value));
        }
        attic.retrieve();
        return remove_const(pos);
    }

    template <typename... Args>
    constexpr iterator emplace(const_iterator pos, Args&&... args)
    {
        capacity_check(size() + 1);
        attic attic{*this, size() + 1, pos};
        emplace_back(std::forward<Args>(args)...);
        attic.retrieve();
        return remove_const(pos);
    }

    template <typename... Args>
    constexpr reference emplace_back(Args&&... args)
    {
        capacity_check(size() + 1);
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
    friend class attic;
    class attic
    {
    public:
        constexpr attic(inplace_vector& v, size_type attic_end, const_iterator pos)
            : v_{v}
            , begin_{attic_end}
            , end_{attic_end}
        {
            if (begin_ == v_.size())
            {
                begin_ = static_cast<size_type>(pos - v_.begin());
                v_.storage_.size(begin_);
            } else {
                for (; v_.end() != pos; --begin_) {
                    const auto v_last = v_.size() - 1;
                    v_.storage_.construct_at(begin_ - 1, std::move(*(v_.begin() + v_last)));
                    v_.storage_.destroy_at(v_last);
                    v_.storage_.size(v_last);
                }
            }
        }
    
        constexpr ~attic()
        {
            v_.storage_.destroy(begin_, end_);
        }

        constexpr void retrieve()
        {
            if (v_.size() == begin_) {
                begin_ = end_;
                v_.storage_.size(end_);
            } else {
                for (; begin_ != end_; ++begin_) {
                    v_.emplace_back(std::move(v_.storage_.data()[begin_]));
                    v_.storage_.destroy_at(begin_);
                }
            }
        }

        constexpr void capacity_check(size_type pos) const
        {
            if (pos >= begin_)
            {
                throw std::bad_alloc{};
            }
        }

    private:
        inplace_vector& v_;
        size_type begin_;
        size_type end_;
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

    constexpr iterator remove_const(const_iterator pos)
    {
        return begin() + (pos - begin());
    }

    [[no_unique_address]] inplace_vector_detail::storage<T, N> storage_;
};

} // namespace jell
