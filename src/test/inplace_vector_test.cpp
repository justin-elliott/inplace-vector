#include "inplace_vector.hpp"

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
static_assert(sizeof(ZeroVector) == sizeof(ZeroVector::non_empty));

namespace {

template <typename T>
class InplaceVectorTest : public testing::Test
{
protected:
    static void append_test_values(T& v, const std::size_t n = T::capacity())
    {
        for (auto& value : test_values(n)) {
            v.emplace_back(std::move(value));
        }
    }

    static bool equal_to_test_values(const T& v, const std::size_t n = T::capacity())
    {
        const auto values = test_values(n);
        return std::equal(v.begin(), v.end(), values.begin(), values.end());
    }

    static auto test_values(const std::size_t n = T::capacity())
    {
        using value_type = typename T::value_type;

        std::vector<value_type> v;
        std::generate_n(std::back_inserter(v), n, [value = 100] mutable {
            return value_type{value++};
        });
        return v;
    }
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
    if constexpr (std::is_copy_constructible_v<typename TypeParam::value_type>) {
        const auto values = this->test_values();

        TypeParam v(values.begin(), values.end());
        EXPECT_THAT(v, ElementsAreArray(values));
    }
}

TYPED_TEST(InplaceVectorTest, is_range_constructible)
{
    if constexpr (std::is_copy_constructible_v<typename TypeParam::value_type>) {
        const auto values = this->test_values();

        TypeParam v(std::from_range, values);
        EXPECT_THAT(v, ElementsAreArray(values));
    }
}

TYPED_TEST(InplaceVectorTest, is_copy_constructible)
{
    if constexpr (std::is_copy_constructible_v<typename TypeParam::value_type>) {
        const auto values = this->test_values();

        TypeParam v1(std::from_range, values);
        TypeParam v2(v1);
        EXPECT_THAT(v2, ElementsAreArray(values));
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
    TypeParam v1;
    this->append_test_values(v1);

    const TypeParam v2(std::move(v1));
    EXPECT_TRUE(this->equal_to_test_values(v2));
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
        const TypeParam v1(std::from_range, this->test_values());

        TypeParam v2;
        v2 = v1;

        EXPECT_EQ(v1, v2);
    }
}

TYPED_TEST(InplaceVectorTest, is_assignable_initially_nonempty_and_smaller)
{
    if constexpr (std::is_copy_assignable_v<typename TypeParam::value_type>) {
        const TypeParam v1(std::from_range, this->test_values());

        TypeParam v2(std::from_range, this->test_values(TypeParam::capacity() / 2));
        v2 = v1;

        EXPECT_EQ(v1, v2);
    }
}

TYPED_TEST(InplaceVectorTest, is_assignable_initially_nonempty_and_larger)
{
    if constexpr (std::is_copy_assignable_v<typename TypeParam::value_type>) {
        const TypeParam v1(std::from_range, this->test_values(TypeParam::capacity() / 2));

        TypeParam v2(std::from_range, this->test_values());
        v2 = v1;

        EXPECT_EQ(v1, v2);
    }
}

TYPED_TEST(InplaceVectorTest, is_move_assignable_initially_empty)
{
    if constexpr (std::is_move_assignable_v<typename TypeParam::value_type>) {
        TypeParam v1;
        this->append_test_values(v1);

        TypeParam v2;
        v2 = std::move(v1);
        EXPECT_TRUE(this->equal_to_test_values(v2));
    }
}

TYPED_TEST(InplaceVectorTest, is_move_assignable_initially_nonempty_and_smaller)
{
    if constexpr (std::is_move_assignable_v<typename TypeParam::value_type>) {
        TypeParam v1;
        this->append_test_values(v1);

        TypeParam v2;
        this->append_test_values(v2, TypeParam::capacity() / 2);

        v2 = std::move(v1);
        EXPECT_TRUE(this->equal_to_test_values(v2));
    }
}

TYPED_TEST(InplaceVectorTest, is_move_assignable_initially_nonempty_and_larger)
{
    if constexpr (std::is_move_assignable_v<typename TypeParam::value_type>) {
        TypeParam v1;
        this->append_test_values(v1, TypeParam::capacity() / 2);

        TypeParam v2;
        this->append_test_values(v2);

        v2 = std::move(v1);
        EXPECT_TRUE(this->equal_to_test_values(v2, TypeParam::capacity() / 2));
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
    if constexpr (std::is_copy_constructible_v<typename TypeParam::value_type>) {
        const auto values = this->test_values();

        TypeParam v;
        v.assign(values.begin(), values.end());
        EXPECT_THAT(v, ElementsAreArray(values));
    }
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
    if constexpr (std::is_copy_constructible_v<typename TypeParam::value_type>) {
        const auto values = this->test_values();

        TypeParam v;
        v.assign_range(values);
        EXPECT_THAT(v, ElementsAreArray(values));
    }
}

TYPED_TEST(InplaceVectorTest, at_in_range)
{
    if constexpr (TypeParam::capacity() != 0) {
        TypeParam v;
        this->append_test_values(v);
        EXPECT_EQ(v.at(0), this->test_values().at(0));
    }
}

TYPED_TEST(InplaceVectorTest, at_out_of_range)
{
    TypeParam v;
    EXPECT_THROW(v.at(0), std::out_of_range);
}

TYPED_TEST(InplaceVectorTest, index_in_range)
{
    if constexpr (TypeParam::capacity() != 0) {
        TypeParam v;
        this->append_test_values(v);
        EXPECT_EQ(v[0], this->test_values().at(0));
    }
}

TYPED_TEST(InplaceVectorTest, front)
{
    if constexpr (TypeParam::capacity() != 0) {
        TypeParam v;
        this->append_test_values(v);
        EXPECT_EQ(v.front(), this->test_values().front());
    }
}

TYPED_TEST(InplaceVectorTest, back)
{
    if constexpr (TypeParam::capacity() != 0) {
        TypeParam v;
        this->append_test_values(v);
        EXPECT_EQ(v.back(), this->test_values().back());
    }
}

TYPED_TEST(InplaceVectorTest, empty)
{
    TypeParam v1;
    EXPECT_TRUE(v1.empty());

    if constexpr (TypeParam::capacity() != 0) {
        v1.emplace_back(typename TypeParam::value_type{100});
        EXPECT_FALSE(v1.empty());
    }
}

TYPED_TEST(InplaceVectorTest, size)
{
    TypeParam v1;
    EXPECT_EQ(v1.size(), 0);

    if constexpr (TypeParam::capacity() != 0) {
        v1.emplace_back(typename TypeParam::value_type{100});
        EXPECT_EQ(v1.size(), 1);
    }
}

TYPED_TEST(InplaceVectorTest, max_size)
{
    TypeParam v1;
    EXPECT_LE(v1.size(), v1.max_size());
}

TYPED_TEST(InplaceVectorTest, capacity)
{
    TypeParam v1;
    EXPECT_LE(v1.size(), v1.capacity());
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
    TypeParam v;
    this->append_test_values(v, TypeParam::capacity() / 2);
    EXPECT_NO_THROW(v.shrink_to_fit());
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
    TypeParam v;
    this->append_test_values(v);

    EXPECT_THROW(v.emplace_back(typename TypeParam::value_type{999}), std::bad_alloc);
}

TYPED_TEST(InplaceVectorTest, can_pop_back)
{
    if constexpr (TypeParam::capacity() != 0) {
        TypeParam v;
        this->append_test_values(v);

        const auto size = v.size();
        v.pop_back();
        EXPECT_EQ(v.size(), size - 1);
    }
}

TYPED_TEST(InplaceVectorTest, can_clear)
{
    TypeParam v;
    this->append_test_values(v);
    EXPECT_EQ(v.size(), v.capacity());

    v.clear();
    EXPECT_EQ(v.size(), 0);
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
        TypeParam v;
        this->append_test_values(v);

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
    TypeParam v;
    EXPECT_THROW(++v.end(), std::range_error);
    EXPECT_THROW(++v.cend(), std::range_error);
#endif
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