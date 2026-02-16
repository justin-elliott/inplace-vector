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

#include "inplace_vector.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

static_assert(std::is_trivially_default_constructible_v<jell::inplace_vector<int, 0>>);
static_assert(jell::inplace_vector<int, 0>{}.size() == 0);
static_assert(jell::inplace_vector<int, 0>::max_size() == 0);
static_assert(jell::inplace_vector<int, 0>::capacity() == 0);
static_assert(jell::inplace_vector<int, 0>{}.data() == nullptr);

static_assert(std::random_access_iterator<jell::inplace_vector<int, 1>::iterator>);
static_assert(std::contiguous_iterator<jell::inplace_vector<int, 1>::iterator>);

struct ZeroVector
{
    int non_empty;
    [[no_unique_address]] jell::inplace_vector<int, 0> empty;
};

#if defined(__INTELLISENSE__)
// 2026-02-14: [[no_unique_address]] is not supported
const auto ZeroVector_expected_size = sizeof(ZeroVector);
#else
const auto ZeroVector_expected_size = sizeof(ZeroVector::non_empty);
#endif // __INTELLISENSE__

static_assert(sizeof(ZeroVector) == ZeroVector_expected_size);

namespace {

template <typename T>
class InplaceVectorTest : public testing::Test
{
protected:
    static constexpr auto make_vector(std::size_t count = T::capacity())
    {
        return std::views::iota(100uz, 100uz + count)
             | std::views::transform([](std::size_t i) { return typename T::value_type{i}; })
             | std::ranges::to<T>();
    }
};

class NonTrivial
{
public:
    constexpr NonTrivial(std::size_t value = 0) : value_{value} {}

    constexpr NonTrivial(const NonTrivial& other) { value_ = other.value_; }
    constexpr NonTrivial(NonTrivial&& other) { value_ = other.value_; }

    constexpr NonTrivial& operator=(const NonTrivial& other) { value_ = other.value_; return *this; }
    constexpr NonTrivial& operator=(NonTrivial&& other) { value_ = other.value_; return *this; }

    constexpr ~NonTrivial() { value_ = 0; }

    constexpr friend bool operator==(const NonTrivial&, const NonTrivial&) = default;

private:
    std::size_t value_;
};

class MoveOnly
{
public:
    constexpr MoveOnly() = default;
    constexpr explicit MoveOnly(std::size_t value) : value_{value} {}

    constexpr MoveOnly(const MoveOnly&) = delete;
    constexpr MoveOnly& operator=(const MoveOnly&) = delete;

    constexpr MoveOnly(MoveOnly&&) = default;
    constexpr MoveOnly& operator=(MoveOnly&&) = default;

    constexpr ~MoveOnly() = default;

    constexpr friend bool operator==(const MoveOnly&, const MoveOnly&) = default;

private:
    std::size_t value_{0};
};

class ThrowOnCopyOrMoveCounter
{
public:
    explicit ThrowOnCopyOrMoveCounter(std::size_t counter) : counter_{counter} {}

    ThrowOnCopyOrMoveCounter(const ThrowOnCopyOrMoveCounter& other) : counter_{other.counter_}
    {
        if (--counter_ <= 0) {
            throw std::runtime_error("Copy Constructor");
        }
    }

    ThrowOnCopyOrMoveCounter(ThrowOnCopyOrMoveCounter&& other) : counter_{other.counter_}
    {
        if (--counter_ <= 0) {
            throw std::runtime_error("Move Constructor");
        }
    }

    ThrowOnCopyOrMoveCounter& operator=(const ThrowOnCopyOrMoveCounter& other)
    {
        counter_ = other.counter_;
        if (--counter_ <= 0) {
            throw std::runtime_error("Copy Assignment");
        }
        return *this;
    }

    ThrowOnCopyOrMoveCounter& operator=(ThrowOnCopyOrMoveCounter&& other)
    {
        counter_ = other.counter_;
        if (--counter_ <= 0) {
            throw std::runtime_error("Move Assignment");
        }
        return *this;
    }

    ~ThrowOnCopyOrMoveCounter() = default;

private:
    std::size_t counter_;
};

template <typename Iter>
class MoveInputIterator
{
public:
    using difference_type   = typename std::move_iterator<Iter>::difference_type;
    using value_type        = typename std::move_iterator<Iter>::value_type;
    using pointer           = typename std::move_iterator<Iter>::pointer;
    using reference         = typename std::move_iterator<Iter>::reference;
    using iterator_category = std::input_iterator_tag;

    explicit MoveInputIterator(Iter iter) : iter_{std::make_move_iterator(iter)} {}

          reference operator*()       { return *iter_; }
    const reference operator*() const { return *iter_; }

    MoveInputIterator& operator++() { ++iter_; }

    MoveInputIterator operator++(int)
    {
        auto tmp(*this);
        ++iter_;
        return tmp;
    }

    friend constexpr bool operator==(const MoveInputIterator&, const MoveInputIterator&) = default;

private:
    std::move_iterator<Iter> iter_;
};

using vector_types = testing::Types<
    jell::inplace_vector<std::size_t, 0>,
    jell::inplace_vector<std::size_t, 23>,
    jell::inplace_vector<NonTrivial, 29>,
    jell::inplace_vector<MoveOnly, 31>
>;
TYPED_TEST_SUITE(InplaceVectorTest, vector_types);

} // namespace

TYPED_TEST(InplaceVectorTest, is_default_constructible)
{
    TypeParam v;
    EXPECT_EQ(v.size(), 0);
}

TYPED_TEST(InplaceVectorTest, is_size_constructible)
{
    constexpr auto count = TypeParam::capacity() / 2;

    TypeParam v(count);
    EXPECT_EQ(v.size(), count);
}

TYPED_TEST(InplaceVectorTest, size_constructor_overflow)
{
    constexpr auto count = TypeParam::capacity() + 1;

    EXPECT_THROW((TypeParam(count)), std::bad_alloc);
}

TYPED_TEST(InplaceVectorTest, is_size_value_constructible)
{
    if constexpr (std::is_copy_constructible_v<typename TypeParam::value_type>) {
        constexpr auto count = TypeParam::capacity() / 2;

        TypeParam v(count, typename TypeParam::value_type{100});
        EXPECT_EQ(v.size(), count);
    }
}

TYPED_TEST(InplaceVectorTest, size_value_constructor_overflow)
{
    if constexpr (std::is_copy_constructible_v<typename TypeParam::value_type>) {
        constexpr auto count = TypeParam::capacity() + 1;

        EXPECT_THROW((TypeParam(count, typename TypeParam::value_type{100})), std::bad_alloc);
    }
}

TYPED_TEST(InplaceVectorTest, is_iterator_constructible)
{
    auto full = this->make_vector();
    const TypeParam v(std::make_move_iterator(full.begin()),
                      std::make_move_iterator(full.end()));
    EXPECT_EQ(v, full);
}

TYPED_TEST(InplaceVectorTest, is_range_constructible)
{
    TypeParam v(std::from_range, this->make_vector() | std::views::as_rvalue);
    EXPECT_EQ(v, this->make_vector());
}

TYPED_TEST(InplaceVectorTest, is_copy_constructible)
{
    if constexpr (std::is_copy_constructible_v<typename TypeParam::value_type>) {
        const auto full = this->make_vector();
        TypeParam v(full);
        EXPECT_EQ(v, full);
    }
}

TEST(InplaceVectorTest, handles_throw_in_copy_constructor)
{
    const std::size_t size = 4;
    using inplace_vector = jell::inplace_vector<ThrowOnCopyOrMoveCounter, size>;

    inplace_vector v(size - 1, ThrowOnCopyOrMoveCounter{3});
    v.emplace_back(ThrowOnCopyOrMoveCounter{2});

    EXPECT_THROW((inplace_vector{v}), std::runtime_error);
}

TEST(InplaceVectorTest, handles_throw_in_move_constructor)
{
    const std::size_t size = 4;
    using inplace_vector = jell::inplace_vector<ThrowOnCopyOrMoveCounter, size>;

    inplace_vector v(size - 1, ThrowOnCopyOrMoveCounter{3});
    v.emplace_back(ThrowOnCopyOrMoveCounter{2});

    EXPECT_THROW((inplace_vector{std::move(v)}), std::runtime_error);
}

TYPED_TEST(InplaceVectorTest, is_move_constructible)
{
    auto full = this->make_vector();
    const TypeParam v(std::move(full));
    EXPECT_EQ(v, this->make_vector());
}

TYPED_TEST(InplaceVectorTest, is_initializer_list_constructible)
{
    using value_type = typename TypeParam::value_type;
    if constexpr (std::is_copy_constructible_v<value_type>) {
        if constexpr (TypeParam::capacity() >= 3) {
            TypeParam v{value_type{100}, value_type{200}, value_type{300}};
            EXPECT_EQ(v.size(), 3);
        }
    }
}

TYPED_TEST(InplaceVectorTest, is_assignable_initially_empty)
{
    if constexpr (std::is_copy_assignable_v<typename TypeParam::value_type>) {
        TypeParam v;
        const auto full = this->make_vector();
        v = full;
        EXPECT_EQ(v, full);
    }
}

TYPED_TEST(InplaceVectorTest, is_assignable_initially_nonempty_and_smaller)
{
    if constexpr (std::is_copy_assignable_v<typename TypeParam::value_type>) {
        const auto full = this->make_vector();
        const auto half = this->make_vector(TypeParam::capacity() / 2);
        TypeParam v(std::from_range, half);
        v = full;
        EXPECT_EQ(v, full);
    }
}

TYPED_TEST(InplaceVectorTest, is_assignable_initially_nonempty_and_larger)
{
    if constexpr (std::is_copy_assignable_v<typename TypeParam::value_type>) {
        const auto full = this->make_vector();
        const auto half = this->make_vector(TypeParam::capacity() / 2);
        TypeParam v(std::from_range, full);
        v = half;
        EXPECT_EQ(v, half);
    }
}

TYPED_TEST(InplaceVectorTest, is_move_assignable_initially_empty)
{
    if constexpr (std::is_move_assignable_v<typename TypeParam::value_type>) {
        TypeParam v;
        auto full = this->make_vector();
        v = std::move(full);
        EXPECT_EQ(v, this->make_vector());
    }
}

TYPED_TEST(InplaceVectorTest, is_move_assignable_initially_nonempty_and_smaller)
{
    if constexpr (std::is_move_assignable_v<typename TypeParam::value_type>) {
        auto full = this->make_vector();
        auto half = this->make_vector(TypeParam::capacity() / 2);
        TypeParam v(std::from_range, half | std::views::as_rvalue);
        v = std::move(full);
        EXPECT_EQ(v, this->make_vector());
    }
}

TYPED_TEST(InplaceVectorTest, is_move_assignable_initially_nonempty_and_larger)
{
    if constexpr (std::is_move_assignable_v<typename TypeParam::value_type>) {
        auto full = this->make_vector();
        auto half = this->make_vector(TypeParam::capacity() / 2);
        TypeParam v(std::from_range, full | std::views::as_rvalue);
        v = std::move(half);
        EXPECT_EQ(v, this->make_vector(TypeParam::capacity() / 2));
    }
}

TYPED_TEST(InplaceVectorTest, can_assign_count_values)
{
    if constexpr (std::is_copy_assignable_v<typename TypeParam::value_type>) {
        const auto value = typename TypeParam::value_type{123};
        const auto count = TypeParam::capacity();
        
        auto v = this->make_vector(count / 2); // Initially non-empty.
        v.assign(count, value);

        EXPECT_EQ(v.size(), count);
        EXPECT_EQ(std::count(v.begin(), v.end(), value), count);
    }
}

TYPED_TEST(InplaceVectorTest, can_assign_iterator)
{
    TypeParam v = this->make_vector(TypeParam::capacity() / 2); // Initially non-empty.
    auto full = this->make_vector();
    v.assign(std::make_move_iterator(full.begin()),
             std::make_move_iterator(full.end()));
    EXPECT_EQ(v, this->make_vector());
}

TYPED_TEST(InplaceVectorTest, can_assign_initializer_list)
{
    using value_type = typename TypeParam::value_type;
    if constexpr (std::is_copy_constructible_v<value_type>) {
        if constexpr (TypeParam::capacity() >= 3) {
            TypeParam v = this->make_vector(); // Initially non-empty.
            v.assign({value_type{100}, value_type{200}, value_type{300}});
            EXPECT_EQ(v.size(), 3);
        }
    }
}

TYPED_TEST(InplaceVectorTest, can_assign_range)
{
    TypeParam v = this->make_vector(); // Initially non-empty.
    v.assign_range(this->make_vector(TypeParam::capacity() / 2) | std::views::as_rvalue);
    EXPECT_EQ(v, this->make_vector(TypeParam::capacity() / 2));
}

TYPED_TEST(InplaceVectorTest, at_in_range)
{
    if constexpr (TypeParam::capacity() != 0) {
        auto v = this->make_vector(1);
        const auto cv = this->make_vector(1);
        EXPECT_EQ(v.at(0), *v.begin());
        EXPECT_EQ(cv.at(0), *cv.begin());
    }
}

TYPED_TEST(InplaceVectorTest, at_out_of_range)
{
    TypeParam v;
    const TypeParam cv{};
    EXPECT_THROW(v.at(1), std::out_of_range);
    EXPECT_THROW(cv.at(1), std::out_of_range);
}

TYPED_TEST(InplaceVectorTest, index_in_range)
{
    if constexpr (TypeParam::capacity() != 0) {
        auto v = this->make_vector(1);
        const auto cv = this->make_vector(1);
        EXPECT_EQ(v[0], *v.begin());
        EXPECT_EQ(cv[0], *cv.begin());
    }
}

TYPED_TEST(InplaceVectorTest, front)
{
    if constexpr (TypeParam::capacity() != 0) {
        auto v = this->make_vector(1);
        const auto cv = this->make_vector(1);
        EXPECT_EQ(v.front(), *v.begin());
        EXPECT_EQ(cv.front(), *cv.begin());
    }
}

TYPED_TEST(InplaceVectorTest, back)
{
    if constexpr (TypeParam::capacity() != 0) {
        auto v = this->make_vector(1);
        const auto cv = this->make_vector(1);
        EXPECT_EQ(v.back(), *(v.end() - 1));
        EXPECT_EQ(cv.back(), *(cv.end() - 1));
    }
}

TYPED_TEST(InplaceVectorTest, empty)
{
    TypeParam v;
    EXPECT_TRUE(v.empty());
}

TYPED_TEST(InplaceVectorTest, nonempty)
{
    if constexpr (TypeParam::capacity() != 0) {
        EXPECT_FALSE(this->make_vector(1).empty());
    }
}

TYPED_TEST(InplaceVectorTest, size_empty)
{
    TypeParam v;
    EXPECT_EQ(v.size(), 0);
}

TYPED_TEST(InplaceVectorTest, size_nonempty)
{
    if constexpr (TypeParam::capacity() != 0) {
        EXPECT_GT(this->make_vector().size(), 0);
    }
}

TYPED_TEST(InplaceVectorTest, max_size)
{
    if constexpr (TypeParam::max_size() != 0) {
        const auto v = this->make_vector(TypeParam::capacity() / 2);
        EXPECT_LT(v.size(), v.max_size());
    }
}

TYPED_TEST(InplaceVectorTest, capacity)
{
    if constexpr (TypeParam::capacity() != 0) {
        const auto v = this->make_vector(TypeParam::capacity() / 2);
        EXPECT_LT(v.size(), v.capacity());
    }
}

TYPED_TEST(InplaceVectorTest, can_resize_smaller)
{
    constexpr auto count = TypeParam::capacity();

    TypeParam v(count);
    v.resize(count / 2);
    EXPECT_EQ(v.size(), count / 2);
}

TYPED_TEST(InplaceVectorTest, can_resize_larger)
{
    constexpr auto count = TypeParam::capacity() / 2;

    TypeParam v(count);
    v.resize(count * 2);
    EXPECT_EQ(v.size(), count * 2);
}

TYPED_TEST(InplaceVectorTest, can_resize_value_smaller)
{
    if constexpr (std::is_copy_constructible_v<typename TypeParam::value_type>) {
        constexpr auto count = TypeParam::capacity();

        TypeParam v(count);
        v.resize(count / 2, typename TypeParam::value_type{100});
        EXPECT_EQ(v.size(), count / 2);
    }
}

TYPED_TEST(InplaceVectorTest, can_resize_value_larger)
{
    if constexpr (std::is_copy_constructible_v<typename TypeParam::value_type>) {
        constexpr auto count = TypeParam::capacity() / 2;

        TypeParam v(count);
        v.resize(count * 2, typename TypeParam::value_type{100});
        EXPECT_EQ(v.size(), count * 2);
    }
}

TYPED_TEST(InplaceVectorTest, reserve_below_capacity)
{
    TypeParam v;
    v.reserve(TypeParam::capacity() / 2);
    EXPECT_EQ(v.capacity(), TypeParam::capacity()); // reserve() has no effect
}

TYPED_TEST(InplaceVectorTest, reserve_above_capacity)
{
    TypeParam v;
    EXPECT_THROW(v.reserve(TypeParam::capacity() + 1), std::bad_alloc);
}

TYPED_TEST(InplaceVectorTest, shrink_to_fit)
{
    auto v = this->make_vector(TypeParam::capacity() / 2);
    EXPECT_NO_THROW(v.shrink_to_fit());
}

TYPED_TEST(InplaceVectorTest, can_insert)
{
    if constexpr (TypeParam::capacity() != 0 && std::is_copy_constructible_v<typename TypeParam::value_type>) {
        const auto half = this->make_vector(TypeParam::capacity() / 2);
        const auto value = typename TypeParam::value_type{200};

        TypeParam v(half);
        const auto starting_size = v.size();
        const auto pos = v.insert(v.begin(), value);
        EXPECT_EQ(pos, v.begin());
        EXPECT_EQ(v.size(), starting_size + 1);
        EXPECT_EQ(v.front(), value);
        EXPECT_TRUE(std::equal(v.begin() + 1, v.end(), half.begin(), half.end()));
    }
}

TYPED_TEST(InplaceVectorTest, can_insert_at_end)
{
    if constexpr (TypeParam::capacity() != 0 && std::is_copy_constructible_v<typename TypeParam::value_type>) {
        const auto half = this->make_vector(TypeParam::capacity() / 2);
        const auto value = typename TypeParam::value_type{200};

        TypeParam v(half);
        const auto starting_size = v.size();
        const auto pos = v.insert(v.end(), value);
        EXPECT_EQ(pos, v.end() - 1);
        EXPECT_EQ(v.size(), starting_size + 1);
        EXPECT_EQ(v.back(), value);
        EXPECT_TRUE(std::equal(v.begin(), v.end() - 1, half.begin(), half.end()));
    }
}

TYPED_TEST(InplaceVectorTest, handles_insert_overflow)
{
    if constexpr (TypeParam::capacity() != 0 && std::is_copy_constructible_v<typename TypeParam::value_type>) {
        TypeParam v(this->make_vector());
        const auto value = typename TypeParam::value_type{999};
        EXPECT_THROW(v.insert(v.begin(), value), std::bad_alloc);
    }
}

TYPED_TEST(InplaceVectorTest, can_move_insert)
{
    if constexpr (TypeParam::capacity() != 0) {
        auto half = this->make_vector(TypeParam::capacity() / 2);
        const auto chalf = this->make_vector(TypeParam::capacity() / 2);
        auto value = typename TypeParam::value_type{200};
        const auto cvalue = typename TypeParam::value_type{200};

        TypeParam v(std::move(half));
        const auto starting_size = v.size();
        const auto pos = v.insert(v.begin(), std::move(value));
        EXPECT_EQ(pos, v.begin());
        EXPECT_EQ(v.size(), starting_size + 1);
        EXPECT_EQ(v.front(), cvalue);
        EXPECT_TRUE(std::equal(v.begin() + 1, v.end(), chalf.begin(), chalf.end()));
    }
}

TYPED_TEST(InplaceVectorTest, can_move_insert_at_end)
{
    if constexpr (TypeParam::capacity() != 0) {
        auto half = this->make_vector(TypeParam::capacity() / 2);
        const auto chalf = this->make_vector(TypeParam::capacity() / 2);
        auto value = typename TypeParam::value_type{200};
        const auto cvalue = typename TypeParam::value_type{200};

        TypeParam v(std::move(half));
        const auto starting_size = v.size();
        const auto pos = v.insert(v.end(), std::move(value));
        EXPECT_EQ(pos, v.end() - 1);
        EXPECT_EQ(v.size(), starting_size + 1);
        EXPECT_EQ(v.back(), cvalue);
        EXPECT_TRUE(std::equal(v.begin(), v.end() - 1, chalf.begin(), chalf.end()));
    }
}

TYPED_TEST(InplaceVectorTest, handles_move_insert_overflow)
{
    if constexpr (TypeParam::capacity() != 0 && std::is_copy_constructible_v<typename TypeParam::value_type>) {
        TypeParam v(this->make_vector());
        auto value = typename TypeParam::value_type{999};
        EXPECT_THROW(v.insert(v.begin(), std::move(value)), std::bad_alloc);
    }
}

TYPED_TEST(InplaceVectorTest, can_count_insert)
{
    if constexpr (TypeParam::capacity() != 0 && std::is_copy_constructible_v<typename TypeParam::value_type>) {
        const auto half = this->make_vector(TypeParam::capacity() / 2);
        const auto count = half.size();
        const auto value = typename TypeParam::value_type{200};

        TypeParam v(half);
        const auto starting_size = v.size();
        const auto pos = v.insert(v.begin(), count, value);
        EXPECT_EQ(pos, v.begin());
        EXPECT_EQ(v.size(), starting_size + count);
        EXPECT_TRUE(std::equal(v.begin() + count, v.end(), half.begin(), half.end()));

        const TypeParam expected(count, value);
        EXPECT_TRUE(std::equal(v.begin(), v.begin() + count, expected.begin(), expected.end()));
    }
}

TYPED_TEST(InplaceVectorTest, can_count_insert_at_end)
{
    if constexpr (TypeParam::capacity() != 0 && std::is_copy_constructible_v<typename TypeParam::value_type>) {
        const auto half = this->make_vector(TypeParam::capacity() / 2);
        const auto count = half.size();
        const auto value = typename TypeParam::value_type{200};

        TypeParam v(half);
        const auto starting_size = v.size();
        const auto pos = v.insert(v.end(), count, value);
        EXPECT_EQ(pos, v.end() - count);
        EXPECT_EQ(v.size(), starting_size + count);
        EXPECT_TRUE(std::equal(v.begin(), pos, half.begin(), half.end()));

        const TypeParam expected(count, value);
        EXPECT_TRUE(std::equal(pos, v.end(), expected.begin(), expected.end()));
    }
}

TYPED_TEST(InplaceVectorTest, handles_count_insert_overflow)
{
    if constexpr (TypeParam::capacity() != 0 && std::is_copy_constructible_v<typename TypeParam::value_type>) {
        TypeParam v(this->make_vector());
        auto value = typename TypeParam::value_type{999};
        EXPECT_THROW(v.insert(v.begin(), 1, value), std::bad_alloc);
    }
}

TYPED_TEST(InplaceVectorTest, can_iterator_insert)
{
    if constexpr (TypeParam::capacity() != 0) {
        TypeParam v(this->make_vector(TypeParam::capacity() / 2));
        const auto starting_size = v.size();

        auto half = this->make_vector(TypeParam::capacity() / 2);
        const auto count = half.size();
        const auto first = std::make_move_iterator(half.begin());
        const auto last  = std::make_move_iterator(half.end());
        const auto pos = v.insert(v.begin(), first, last);
        EXPECT_EQ(pos, v.begin());
        EXPECT_EQ(v.size(), starting_size + count);
        EXPECT_TRUE(std::equal(v.begin() + count, v.end(), half.begin(), half.end()));

        const TypeParam expected(this->make_vector(TypeParam::capacity() / 2));
        EXPECT_TRUE(std::equal(v.begin(), v.begin() + count, expected.begin(), expected.end()));
    }
}

TYPED_TEST(InplaceVectorTest, can_iterator_insert_without_random_access)
{
    if constexpr (TypeParam::capacity() != 0) {
        TypeParam v(this->make_vector(TypeParam::capacity() / 2));
        const auto starting_size = v.size();

        auto half = this->make_vector(TypeParam::capacity() / 2);
        const auto count = half.size();
        const auto first = MoveInputIterator{half.begin()};
        const auto last  = MoveInputIterator{half.end()};
        const auto pos = v.insert(v.begin(), first, last);
        EXPECT_EQ(pos, v.begin());
        EXPECT_EQ(v.size(), starting_size + count);
        EXPECT_TRUE(std::equal(v.begin() + count, v.end(), half.begin(), half.end()));

        const TypeParam expected(this->make_vector(TypeParam::capacity() / 2));
        EXPECT_TRUE(std::equal(v.begin(), v.begin() + count, expected.begin(), expected.end()));
    }
}

TYPED_TEST(InplaceVectorTest, can_iterator_insert_at_end)
{
    if constexpr (TypeParam::capacity() != 0) {
        TypeParam v(this->make_vector(TypeParam::capacity() / 2));
        const auto starting_size = v.size();

        auto half = this->make_vector(TypeParam::capacity() / 2);
        const auto count = half.size();
        const auto first = std::make_move_iterator(half.begin());
        const auto last  = std::make_move_iterator(half.end());
        const auto pos = v.insert(v.end(), first, last);
        EXPECT_EQ(pos, v.end() - count);
        EXPECT_EQ(v.size(), starting_size + count);
        EXPECT_TRUE(std::equal(v.begin(), pos, half.begin(), half.end()));

        const TypeParam expected(this->make_vector(TypeParam::capacity() / 2));
        EXPECT_TRUE(std::equal(pos, v.end(), expected.begin(), expected.end()));
    }
}

TYPED_TEST(InplaceVectorTest, handles_iterator_insert_overflow)
{
    if constexpr (TypeParam::capacity() != 0) {
        TypeParam v(this->make_vector());
        TypeParam v_insert(this->make_vector(1));
        EXPECT_THROW(v.insert(v.begin(),
                              std::make_move_iterator(v_insert.begin()),
                              std::make_move_iterator(v_insert.end())),
                     std::bad_alloc);
    }
}

TYPED_TEST(InplaceVectorTest, handles_iterator_insert_overflow_without_random_access)
{
    if constexpr (TypeParam::capacity() != 0) {
        TypeParam v(this->make_vector());
        TypeParam v_insert(this->make_vector(1));
        const auto first = MoveInputIterator{v_insert.begin()};
        const auto last  = MoveInputIterator{v_insert.end()};
        EXPECT_THROW(v.insert(v.begin(), first, last), std::bad_alloc);
    }
}

TYPED_TEST(InplaceVectorTest, can_insert_initializer_list)
{
    using value_type = typename TypeParam::value_type;
    if constexpr (std::is_copy_constructible_v<value_type>) {
        if constexpr (TypeParam::capacity() >= 3) {
            TypeParam v(1, value_type{300});
            v.insert(v.begin(), {value_type{100}, value_type{200}});
            EXPECT_EQ(v.size(), 3);
            EXPECT_EQ(v.at(0), value_type{100});
            EXPECT_EQ(v.at(1), value_type{200});
            EXPECT_EQ(v.at(2), value_type{300});
        }
    }
}

TYPED_TEST(InplaceVectorTest, can_insert_range)
{
    TypeParam v(this->make_vector(TypeParam::capacity() / 2));
    const auto pos = v.insert_range(v.begin(), this->make_vector(TypeParam::capacity() / 2) | std::views::as_rvalue);
    EXPECT_EQ(pos, v.begin());

    TypeParam half(this->make_vector(TypeParam::capacity() / 2));
    EXPECT_TRUE(std::equal(v.begin(), v.begin() + half.size(), half.begin(), half.end()));
    EXPECT_TRUE(std::equal(v.begin() + half.size(), v.end(), half.begin(), half.end()));
}

TYPED_TEST(InplaceVectorTest, handles_insert_range_overflow)
{
    if constexpr (TypeParam::capacity() != 0) {
        TypeParam v(this->make_vector());
        EXPECT_THROW(v.insert_range(v.begin(), this->make_vector(1) | std::views::as_rvalue), std::bad_alloc);
    }
}

TYPED_TEST(InplaceVectorTest, can_emplace)
{
    if constexpr (TypeParam::capacity() != 0 && std::is_copy_constructible_v<typename TypeParam::value_type>) {
        const auto half = this->make_vector(TypeParam::capacity() / 2);
        const std::size_t raw_value = 200;
        const auto value = typename TypeParam::value_type{raw_value};

        TypeParam v(half);
        const auto starting_size = v.size();
        const auto pos = v.emplace(v.begin(), raw_value);
        EXPECT_EQ(pos, v.begin());
        EXPECT_EQ(v.size(), starting_size + 1);
        EXPECT_EQ(v.front(), value);
        EXPECT_TRUE(std::equal(v.begin() + 1, v.end(), half.begin(), half.end()));
    }
}

TYPED_TEST(InplaceVectorTest, can_emplace_at_end)
{
    if constexpr (TypeParam::capacity() != 0 && std::is_copy_constructible_v<typename TypeParam::value_type>) {
        const auto half = this->make_vector(TypeParam::capacity() / 2);
        const std::size_t raw_value = 200;
        const auto value = typename TypeParam::value_type{raw_value};

        TypeParam v(half);
        const auto starting_size = v.size();
        const auto pos = v.emplace(v.end(), raw_value);
        EXPECT_EQ(pos, v.end() - 1);
        EXPECT_EQ(v.size(), starting_size + 1);
        EXPECT_EQ(v.back(), value);
        EXPECT_TRUE(std::equal(v.begin(), v.end() - 1, half.begin(), half.end()));
    }
}

TYPED_TEST(InplaceVectorTest, handles_emplace_overflow)
{
    if constexpr (TypeParam::capacity() != 0 && std::is_copy_constructible_v<typename TypeParam::value_type>) {
        TypeParam v(this->make_vector());
        EXPECT_THROW(v.emplace(v.begin(), 999), std::bad_alloc);
    }
}

TYPED_TEST(InplaceVectorTest, can_emplace_back)
{
    if constexpr (TypeParam::capacity() != 0) {
        TypeParam v;
        const std::size_t raw_value = 123;
        const auto value = typename TypeParam::value_type{raw_value};
        v.emplace_back(raw_value);
        EXPECT_EQ(v.at(0), value);
    }
}

TYPED_TEST(InplaceVectorTest, handles_emplace_back_overflow)
{
    auto full = this->make_vector();
    EXPECT_THROW(full.emplace_back(typename TypeParam::value_type{999}), std::bad_alloc);
}

TYPED_TEST(InplaceVectorTest, can_try_emplace_back)
{
    if constexpr (TypeParam::capacity() != 0) {
        TypeParam v;
        const std::size_t raw_value = 123;
        const auto value = typename TypeParam::value_type{raw_value};
        EXPECT_NE(v.try_emplace_back(raw_value), nullptr);
        EXPECT_EQ(v.at(0), value);
    }
}

TYPED_TEST(InplaceVectorTest, handles_try_emplace_back_overflow)
{
    auto full = this->make_vector();
    EXPECT_EQ(full.try_emplace_back(typename TypeParam::value_type{999}), nullptr);
}

TYPED_TEST(InplaceVectorTest, can_push_back)
{
    if constexpr (TypeParam::capacity() != 0 && std::is_copy_constructible_v<typename TypeParam::value_type>) {
        TypeParam v;
        const auto value = typename TypeParam::value_type{123};
        v.push_back(value);
        EXPECT_EQ(v.at(0), value);
    }
}

TYPED_TEST(InplaceVectorTest, can_push_back_rvalue)
{
    if constexpr (TypeParam::capacity() != 0) {
        TypeParam v;
        auto value = typename TypeParam::value_type{123};
        const auto cvalue = typename TypeParam::value_type{123};
        v.push_back(std::move(value));
        EXPECT_EQ(v.at(0), cvalue);
    }
}

TYPED_TEST(InplaceVectorTest, can_try_push_back)
{
    if constexpr (TypeParam::capacity() != 0 && std::is_copy_constructible_v<typename TypeParam::value_type>) {
        TypeParam v;
        const auto value = typename TypeParam::value_type{123};
        EXPECT_NE(v.try_push_back(value), nullptr);
        EXPECT_EQ(v.at(0), value);
    }
}

TYPED_TEST(InplaceVectorTest, can_try_push_back_rvalue)
{
    if constexpr (TypeParam::capacity() != 0) {
        TypeParam v;
        auto value = typename TypeParam::value_type{123};
        const auto cvalue = typename TypeParam::value_type{123};
        EXPECT_NE(v.try_push_back(std::move(value)), nullptr);
        EXPECT_EQ(v.at(0), cvalue);
    }
}

TYPED_TEST(InplaceVectorTest, can_pop_back)
{
    if constexpr (TypeParam::capacity() != 0) {
        auto v = this->make_vector();
        const auto starting_size = v.size();
        v.pop_back();
        EXPECT_EQ(v.size(), starting_size - 1);
    }
}

TYPED_TEST(InplaceVectorTest, can_append_range)
{
    TypeParam v(this->make_vector(TypeParam::capacity() / 2));
    v.append_range(this->make_vector(TypeParam::capacity() / 2) | std::views::as_rvalue);

    TypeParam half(this->make_vector(TypeParam::capacity() / 2));
    EXPECT_TRUE(std::equal(v.begin(), v.begin() + half.size(), half.begin(), half.end()));
    EXPECT_TRUE(std::equal(v.begin() + half.size(), v.end(), half.begin(), half.end()));
}

TYPED_TEST(InplaceVectorTest, handles_append_range_overflow)
{
    if constexpr (TypeParam::capacity() != 0) {
        TypeParam v(this->make_vector());
        EXPECT_THROW(v.append_range(this->make_vector(1) | std::views::as_rvalue), std::bad_alloc);
    }
}

TYPED_TEST(InplaceVectorTest, can_try_append_range)
{
    auto v = this->make_vector(TypeParam::capacity() / 2);
    auto append = this->make_vector(TypeParam::capacity() / 2) | std::views::as_rvalue;
    const auto append_end = std::ranges::end(append);
    EXPECT_EQ(v.try_append_range(append), append_end);

    TypeParam half(this->make_vector(TypeParam::capacity() / 2));
    EXPECT_TRUE(std::equal(v.begin(), v.begin() + half.size(), half.begin(), half.end()));
    EXPECT_TRUE(std::equal(v.begin() + half.size(), v.end(), half.begin(), half.end()));
}

TYPED_TEST(InplaceVectorTest, handles_try_append_range_overflow)
{
    if constexpr (TypeParam::capacity() != 0) {
        auto v = this->make_vector(TypeParam::capacity() / 2);
        auto append = this->make_vector() | std::views::as_rvalue;
        const auto append_end = std::ranges::end(append);
        EXPECT_NE(v.try_append_range(append), append_end);
        EXPECT_EQ(v.size(), v.capacity());
    }
}

TYPED_TEST(InplaceVectorTest, can_clear)
{
    auto v = this->make_vector();
    v.clear();
    EXPECT_EQ(v.size(), 0);
}

TYPED_TEST(InplaceVectorTest, can_erase_single)
{
    if constexpr (TypeParam::capacity() != 0) {
        auto v = this->make_vector();
        const auto starting_size = v.size();
        const auto result_iter = v.erase(v.begin());
        EXPECT_EQ(result_iter, v.begin());
        EXPECT_EQ(v.size(), starting_size - 1);
    }
}

TYPED_TEST(InplaceVectorTest, can_erase_single_last)
{
    if constexpr (TypeParam::capacity() != 0) {
        auto v = this->make_vector();
        const auto starting_size = v.size();
        const auto result_iter = v.erase(v.end() - 1);
        EXPECT_EQ(result_iter, v.end());
        EXPECT_EQ(v.size(), starting_size - 1);
    }
}

TYPED_TEST(InplaceVectorTest, can_erase)
{
    if constexpr (TypeParam::capacity() != 0) {
        auto v = this->make_vector();
        const auto starting_size = v.size();
        const auto result_iter = v.erase(v.begin(), v.begin() + starting_size / 2);
        EXPECT_EQ(result_iter, v.begin());
        EXPECT_EQ(v.size(), starting_size - starting_size / 2);

        const auto expected = this->make_vector();
        EXPECT_TRUE(std::equal(v.begin(), v.end(), expected.begin() + starting_size / 2, expected.end()));
    }
}

TYPED_TEST(InplaceVectorTest, can_erase_last)
{
    if constexpr (TypeParam::capacity() != 0) {
        auto v = this->make_vector();
        const auto starting_size = v.size();
        const auto result_iter = v.erase(v.begin() + starting_size / 2, v.end());
        EXPECT_EQ(result_iter, v.end());
        EXPECT_EQ(v.size(), starting_size / 2);

        const auto expected = this->make_vector();
        EXPECT_TRUE(std::equal(v.begin(), v.end(), expected.begin(), expected.begin() + starting_size / 2));
    }
}

TYPED_TEST(InplaceVectorTest, can_swap_same_size)
{
    const auto expected_lhs = this->make_vector();
    const auto expected_rhs = this->make_vector();

    auto lhs = this->make_vector();
    auto rhs = this->make_vector();
    std::swap(lhs, rhs);

    EXPECT_EQ(lhs, expected_lhs);
    EXPECT_EQ(rhs, expected_rhs);
}

TYPED_TEST(InplaceVectorTest, can_swap_lhs_smaller)
{
    const auto expected_lhs = this->make_vector();
    const auto expected_rhs = this->make_vector(TypeParam::capacity() / 2);

    auto lhs = this->make_vector(TypeParam::capacity() / 2);
    auto rhs = this->make_vector();
    std::swap(lhs, rhs);

    EXPECT_EQ(lhs, expected_lhs);
    EXPECT_EQ(rhs, expected_rhs);
}

TYPED_TEST(InplaceVectorTest, can_swap_lhs_greater)
{
    const auto expected_lhs = this->make_vector(TypeParam::capacity() / 2);
    const auto expected_rhs = this->make_vector();

    auto lhs = this->make_vector();
    auto rhs = this->make_vector(TypeParam::capacity() / 2);
    std::swap(lhs, rhs);

    EXPECT_EQ(lhs, expected_lhs);
    EXPECT_EQ(rhs, expected_rhs);
}

TYPED_TEST(InplaceVectorTest, can_erase_value)
{
    using value_type = typename TypeParam::value_type;

    TypeParam v;
    for (std::size_t i = 0; i != v.capacity(); ++i) {
        v.emplace_back(100 + (i & 1));
    }

    std::erase(v, value_type{100});

    const auto expected_value = value_type{101};
    EXPECT_EQ(std::count(v.begin(), v.end(), expected_value), v.size());
}

TYPED_TEST(InplaceVectorTest, can_erase_predicate)
{
    using value_type = typename TypeParam::value_type;

    TypeParam v;
    for (std::size_t i = 0; i != v.capacity(); ++i) {
        v.emplace_back(100 + (i & 1));
    }

    std::erase_if(v, [i = 0](const value_type&) mutable { return (i++ & 1) == 0; });

    const auto expected_value = value_type{101};
    EXPECT_EQ(std::count(v.begin(), v.end(), expected_value), v.size());
}

TYPED_TEST(InplaceVectorTest, can_compare_iterators)
{
    TypeParam v;
    EXPECT_EQ(v.begin(), v.begin());
    EXPECT_EQ(v.begin(), v.cbegin());
    EXPECT_EQ(v.end(), v.end());
    EXPECT_EQ(v.end(), v.cend());

    EXPECT_LE(v.begin(), v.end());

    typename TypeParam::const_iterator ci;
    ci = v.cbegin();
    EXPECT_EQ(ci, v.begin());
}

TYPED_TEST(InplaceVectorTest, can_dereference_iterators)
{
    if constexpr (TypeParam::capacity() != 0) {
        auto v = this->make_vector(1);

        EXPECT_EQ(*v.begin(), v.at(0));
        EXPECT_EQ(*v.cbegin(), v.at(0));
    }
}

TYPED_TEST(InplaceVectorTest, cant_dereference_iterators_past_end)
{
#if defined(CHECKED_ITERATORS)
    if constexpr (TypeParam::capacity() != 0) {
        auto v = this->make_vector(1);
        EXPECT_THROW(*v.end(), std::range_error);
        EXPECT_THROW(*v.cend(), std::range_error);
    }
#endif
}

TYPED_TEST(InplaceVectorTest, cant_increment_iterators_past_end)
{
#if defined(CHECKED_ITERATORS)
    TypeParam v;
    EXPECT_THROW(++v.end(), std::range_error);
    EXPECT_THROW(++v.cend(), std::range_error);
#endif
}

TYPED_TEST(InplaceVectorTest, can_use_addition_on_iterators)
{
    if constexpr (TypeParam::capacity() > 2) {
        auto v = this->make_vector();

        EXPECT_EQ(*(v.begin() + 2), v.at(2));
        EXPECT_EQ(*(v.cbegin() + 2), v.at(2));
        
        auto iter = v.begin();
        iter += 2;
        auto const_iter = v.cbegin();
        const_iter += 2;

        EXPECT_EQ(*iter, v.at(2));
        EXPECT_EQ(*const_iter, v.at(2));
    }
}

TYPED_TEST(InplaceVectorTest, can_use_subtraction_on_iterators)
{
    if constexpr (TypeParam::capacity() > 2) {
        auto v = this->make_vector();

        EXPECT_EQ(*(v.end() - 2), v.at(v.size() - 2));
        EXPECT_EQ(*(v.cend() - 2), v.at(v.size() - 2));
        EXPECT_EQ(v.end() - v.begin(), v.size());
        
        auto iter = v.end();
        iter -= 2;
        auto const_iter = v.cend();
        const_iter -= 2;

        EXPECT_EQ(*iter, v.at(v.size() - 2));
        EXPECT_EQ(*const_iter, v.at(v.size() - 2));
    }
}

TEST(StorageTest, can_call_members_in_zero_size)
{
    using storage_type = jell::detail::inplace_vector::storage<int, 0>;
    storage_type storage;

    EXPECT_EQ(storage.data(), nullptr);
    EXPECT_EQ(const_cast<const storage_type&>(storage).data(), nullptr);
    EXPECT_EQ(storage.size(), 0);
    EXPECT_EQ(const_cast<const storage_type&>(storage).size(), 0);

    EXPECT_NO_THROW(storage.size(0));
    EXPECT_NO_THROW(storage.construct_at(0, 100));
    EXPECT_NO_THROW(storage.destroy_at(0));
    EXPECT_NO_THROW(storage.destroy(0, 0));
    EXPECT_NO_THROW(storage.clear());
}
