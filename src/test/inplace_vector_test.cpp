#include "inplace_vector.hpp"

#include <array>
#include <limits>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using testing::ElementsAreArray;

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

constexpr auto at_end = std::numeric_limits<std::size_t>::max();

constexpr auto move_begin(auto& container)
{
    return std::make_move_iterator(container.begin());
}

constexpr auto move_end(auto& container, std::size_t n = at_end)
{
    return std::make_move_iterator((n == at_end) ? container.end() : container.begin() + n);
}

template <typename T, typename IntegerSequence = std::make_integer_sequence<int, T::capacity()>>
struct vectors;

template <typename T, int... Is>
class vectors<T, std::integer_sequence<int, Is...>>
{
private:
    using mutable_vector = T;
    using const_vector = const T;
    using value_type = mutable_vector::value_type;

    static constexpr inline auto capacity = mutable_vector::capacity();

    std::array<value_type, capacity> a_full    {value_type{Is + 100}...};
    std::array<value_type, capacity> a_full_mut{value_type{Is + 100}...};
    std::array<value_type, capacity> a_half    {value_type{Is + 200}...};
    std::array<value_type, capacity> a_half_mut{value_type{Is + 200}...};

public:
    constexpr vectors()
        : v_full    (move_begin(a_full),     move_end(a_full))
        , v_full_mut(move_begin(a_full_mut), move_end(a_full_mut))
        , v_half    (move_begin(a_half),     move_end(a_half,     capacity / 2))
        , v_half_mut(move_begin(a_half_mut), move_end(a_half_mut, capacity / 2))
    {
    }

    const_vector   v_full;
    mutable_vector v_full_mut;
    const_vector   v_half;
    mutable_vector v_half_mut;
};

template <typename T>
class InplaceVectorTest : public testing::Test, public vectors<T>
{
};

class NonTrivial
{
public:
    constexpr NonTrivial(int value = 0) : value_{value} {}

    constexpr NonTrivial(const NonTrivial& other) { value_ = other.value_; }
    constexpr NonTrivial(NonTrivial&& other) { value_ = other.value_; }

    constexpr NonTrivial& operator=(const NonTrivial& other) { value_ = other.value_; return *this; }
    constexpr NonTrivial& operator=(NonTrivial&& other) { value_ = other.value_; return *this; }

    constexpr ~NonTrivial() { value_ = -1; }

    constexpr friend bool operator==(const NonTrivial& lhs, const NonTrivial& rhs)
    {
        return lhs.value_ == rhs.value_;
    }

private:
    int value_;
};

class MoveOnly
{
public:
    constexpr MoveOnly() = default;
    constexpr explicit MoveOnly(int value) : value_{value} {}

    constexpr MoveOnly(const MoveOnly&) = delete;
    constexpr MoveOnly& operator=(const MoveOnly&) = delete;

    constexpr MoveOnly(MoveOnly&&) = default;
    constexpr MoveOnly& operator=(MoveOnly&&) = default;

    constexpr ~MoveOnly() = default;

    constexpr friend bool operator==(const MoveOnly& lhs, const MoveOnly& rhs)
    {
        return lhs.value_ == rhs.value_;
    }

private:
    int value_{0};
};

class ThrowOnCopyOrMoveCounter
{
public:
    explicit ThrowOnCopyOrMoveCounter(int counter) : counter_{counter} {}

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
    int counter_;
};

using vector_types = testing::Types<
    jell::inplace_vector<int, 32>,
    jell::inplace_vector<int, 0>,
    jell::inplace_vector<NonTrivial, 24>,
    jell::inplace_vector<MoveOnly, 16>
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
    const TypeParam v(move_begin(this->v_full_mut), move_end(this->v_full_mut));
    EXPECT_EQ(v, this->v_full);
}

TYPED_TEST(InplaceVectorTest, is_range_constructible)
{
    TypeParam v(std::from_range, this->v_full_mut | std::views::as_rvalue);
    EXPECT_EQ(v, this->v_full);
}

TYPED_TEST(InplaceVectorTest, is_copy_constructible)
{
    if constexpr (std::is_copy_constructible_v<typename TypeParam::value_type>) {
        TypeParam v(this->v_full);
        EXPECT_EQ(v, this->v_full);
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
    const TypeParam v(std::move(this->v_full_mut));
    EXPECT_EQ(v, this->v_full);
}

TYPED_TEST(InplaceVectorTest, is_initializer_list_constructible)
{
    using value_type = typename TypeParam::value_type;
    if constexpr (std::is_copy_constructible_v<value_type>) {
        if constexpr (TypeParam::capacity() >= 3) {
            TypeParam v{value_type{}, value_type{}, value_type{}};
            EXPECT_EQ(v.size(), 3);
        }
    }
}

TYPED_TEST(InplaceVectorTest, is_assignable_initially_empty)
{
    if constexpr (std::is_copy_assignable_v<typename TypeParam::value_type>) {
        TypeParam v;
        v = this->v_full;
        EXPECT_EQ(v, this->v_full);
    }
}

TYPED_TEST(InplaceVectorTest, is_assignable_initially_nonempty_and_smaller)
{
    if constexpr (std::is_copy_assignable_v<typename TypeParam::value_type>) {
        TypeParam v(std::from_range, this->v_half);
        v = this->v_full;
        EXPECT_EQ(v, this->v_full);
    }
}

TYPED_TEST(InplaceVectorTest, is_assignable_initially_nonempty_and_larger)
{
    if constexpr (std::is_copy_assignable_v<typename TypeParam::value_type>) {
        TypeParam v(std::from_range, this->v_full);
        v = this->v_half;
        EXPECT_EQ(v, this->v_half);
    }
}

TYPED_TEST(InplaceVectorTest, is_move_assignable_initially_empty)
{
    if constexpr (std::is_move_assignable_v<typename TypeParam::value_type>) {
        TypeParam v;
        v = std::move(this->v_full_mut);
        EXPECT_EQ(v, this->v_full);
    }
}

TYPED_TEST(InplaceVectorTest, is_move_assignable_initially_nonempty_and_smaller)
{
    if constexpr (std::is_move_assignable_v<typename TypeParam::value_type>) {
        TypeParam v(std::from_range, this->v_half_mut | std::views::as_rvalue);
        v = std::move(this->v_full_mut);
        EXPECT_EQ(v, this->v_full);
    }
}

TYPED_TEST(InplaceVectorTest, is_move_assignable_initially_nonempty_and_larger)
{
    if constexpr (std::is_move_assignable_v<typename TypeParam::value_type>) {
        TypeParam v(std::from_range, this->v_full_mut | std::views::as_rvalue);
        v = std::move(this->v_half_mut);
        EXPECT_EQ(v, this->v_half);
    }
}

TYPED_TEST(InplaceVectorTest, can_assign_count_values)
{
    if constexpr (std::is_copy_assignable_v<typename TypeParam::value_type>) {
        const typename TypeParam::value_type value_1{123};
        const typename TypeParam::value_type value_2{456};
        const auto lower_count = TypeParam::capacity() / 2;
        const auto upper_count = TypeParam::capacity() - lower_count;
        
        TypeParam v;
        v.assign(lower_count, value_1);
        v.assign(upper_count, value_2);

        EXPECT_EQ(v.size(), TypeParam::capacity());
        EXPECT_EQ(std::count(v.begin(), v.end(), value_1), lower_count);
        EXPECT_EQ(std::count(v.begin(), v.end(), value_2), upper_count);
    }
}

TYPED_TEST(InplaceVectorTest, can_assign_iterator)
{
    TypeParam v;
    v.assign(move_begin(this->v_full_mut), move_end(this->v_full_mut));
    EXPECT_EQ(v, this->v_full);
}

TYPED_TEST(InplaceVectorTest, can_assign_initializer_list)
{
    using value_type = typename TypeParam::value_type;
    if constexpr (std::is_copy_constructible_v<value_type>) {
        if constexpr (TypeParam::capacity() >= 3) {
            TypeParam v;
            v.assign({value_type{}, value_type{}, value_type{}});
            EXPECT_EQ(v.size(), 3);
        }
    }
}

TYPED_TEST(InplaceVectorTest, can_assign_range)
{
    TypeParam v;
    v.assign_range(this->v_full_mut | std::views::as_rvalue);
    EXPECT_EQ(v, this->v_full);
}

TYPED_TEST(InplaceVectorTest, at_in_range)
{
    if constexpr (TypeParam::capacity() != 0) {
        EXPECT_NO_THROW(this->v_full.at(0));
        EXPECT_NO_THROW(this->v_full_mut.at(0));
    }
}

TYPED_TEST(InplaceVectorTest, at_out_of_range)
{
    TypeParam v;
    EXPECT_THROW(this->v_full.at(this->v_full.size()), std::out_of_range);
    EXPECT_THROW(this->v_full_mut.at(this->v_full.size()), std::out_of_range);
}

TYPED_TEST(InplaceVectorTest, index_in_range)
{
    if constexpr (TypeParam::capacity() != 0) {
        EXPECT_NO_THROW(this->v_full[0]);
        EXPECT_NO_THROW(this->v_full_mut[0]);
    }
}

TYPED_TEST(InplaceVectorTest, front)
{
    if constexpr (TypeParam::capacity() != 0) {
        EXPECT_EQ(this->v_full.front(), this->v_full[0]);
        EXPECT_EQ(this->v_full_mut.front(), this->v_full_mut[0]);
    }
}

TYPED_TEST(InplaceVectorTest, back)
{
    if constexpr (TypeParam::capacity() != 0) {
        EXPECT_EQ(this->v_full.back(), this->v_full[this->v_full.size() - 1]);
        EXPECT_EQ(this->v_full_mut.back(), this->v_full_mut[this->v_full_mut.size() - 1]);
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
        EXPECT_FALSE(this->v_full.empty());
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
        EXPECT_GT(this->v_half.size(), 0);
    }
}

TYPED_TEST(InplaceVectorTest, max_size)
{
    EXPECT_LE(this->v_half.size(), this->v_half.max_size());
}

TYPED_TEST(InplaceVectorTest, capacity)
{
    EXPECT_LE(this->v_half.size(), this->v_half.capacity());
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
    EXPECT_NO_THROW(this->v_half_mut.shrink_to_fit());
}

TYPED_TEST(InplaceVectorTest, can_emplace_back)
{
    if constexpr (TypeParam::capacity() != 0) {
        TypeParam v;
        v.emplace_back(typename TypeParam::value_type{100});
    }
}

TYPED_TEST(InplaceVectorTest, handles_emplace_back_overflow)
{
    EXPECT_THROW(this->v_full_mut.emplace_back(typename TypeParam::value_type{999}), std::bad_alloc);
}

TYPED_TEST(InplaceVectorTest, can_pop_back)
{
    if constexpr (TypeParam::capacity() != 0) {
        this->v_full_mut.pop_back();
        EXPECT_EQ(this->v_full_mut.size(), this->v_full.size() - 1);
    }
}

TYPED_TEST(InplaceVectorTest, can_clear)
{
    this->v_full_mut.clear();
    EXPECT_EQ(this->v_full_mut.size(), 0);
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
        TypeParam& v = this->v_full_mut;

        EXPECT_EQ(*v.begin(), v[0]);
        EXPECT_EQ(*v.cbegin(), v[0]);

#if defined(CHECKED_ITERATORS)
        EXPECT_THROW(*v.end(), std::range_error);
        EXPECT_THROW(*v.cend(), std::range_error);
#endif
    }
}

TYPED_TEST(InplaceVectorTest, cant_increment_iterators_past_end)
{
#if defined(CHECKED_ITERATORS)
    TypeParam& v = this->v_full_mut;
    EXPECT_THROW(++v.end(), std::range_error);
    EXPECT_THROW(++v.cend(), std::range_error);
#endif
}

TYPED_TEST(InplaceVectorTest, can_use_addition_on_iterators)
{
    if constexpr (TypeParam::capacity() > 2) {
        TypeParam& v = this->v_full_mut;

        EXPECT_EQ(*(v.begin() + 2), v[2]);
        EXPECT_EQ(*(v.cbegin() + 2), v[2]);
        
        auto iter = v.begin();
        iter += 2;
        auto const_iter = v.cbegin();
        const_iter += 2;

        EXPECT_EQ(*iter, v[2]);
        EXPECT_EQ(*const_iter, v[2]);
    }
}

TYPED_TEST(InplaceVectorTest, can_use_subtraction_on_iterators)
{
    if constexpr (TypeParam::capacity() > 2) {
        TypeParam& v = this->v_full_mut;

        EXPECT_EQ(*(v.end() - 2), v[v.size() - 2]);
        EXPECT_EQ(*(v.cend() - 2), v[v.size() - 2]);
        EXPECT_EQ(v.end() - v.begin(), v.size());
        
        auto iter = v.end();
        iter -= 2;
        auto const_iter = v.cend();
        const_iter -= 2;

        EXPECT_EQ(*iter, v[v.size() - 2]);
        EXPECT_EQ(*const_iter, v[v.size() - 2]);
    }
}

TEST(StorageTest, can_call_members_in_zero_size)
{
    using storage = jell::inplace_vector_detail::storage<int, 0>;
    storage s;

    EXPECT_EQ(s.data(), nullptr);
    EXPECT_EQ(const_cast<const storage&>(s).data(), nullptr);
    EXPECT_EQ(s.size(), 0);
    EXPECT_EQ(const_cast<const storage&>(s).size(), 0);

    EXPECT_NO_THROW(s.size(0));
    EXPECT_NO_THROW(s.construct_at(0, 100));
    EXPECT_NO_THROW(s.destroy_at(0));
    EXPECT_NO_THROW(s.destroy(0, 0));
    EXPECT_NO_THROW(s.clear());
}