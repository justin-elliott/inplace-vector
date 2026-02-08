#include "inplace_vector.hpp"

#include <array>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using testing::ElementsAreArray;

static_assert(std::is_trivially_default_constructible_v<jell::inplace_vector<int, 0>>);
static_assert(sizeof(jell::inplace_vector<int, 0>) < sizeof(int));
static_assert(jell::inplace_vector<int, 0>{}.size() == 0);
static_assert(jell::inplace_vector<int, 0>{}.max_size() == 0);
static_assert(jell::inplace_vector<int, 0>{}.capacity() == 0);
static_assert(jell::inplace_vector<int, 0>{}.data() == nullptr);

namespace {

template <typename T>
class InplaceVectorTest : public testing::Test
{
protected:
    static auto test_values()
    {
        using value_type = typename T::value_type;
        constexpr auto size = T{}.max_size();

        std::array<value_type, size> a;
        for (int i = 0; i != static_cast<int>(size); ++i) {
            a[i] = value_type{i};
        }
        return a;
    }
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

using vector_types = testing::Types<
    jell::inplace_vector<int, 32>,
    jell::inplace_vector<int, 0>,
    jell::inplace_vector<MoveOnly, 16>
>;
TYPED_TEST_SUITE(InplaceVectorTest, vector_types);

} // namespace

TYPED_TEST(InplaceVectorTest, isDefaultConstructible)
{
    TypeParam v;
    EXPECT_EQ(v.size(), 0);
}

TYPED_TEST(InplaceVectorTest, isSizeConstructible)
{
    constexpr auto count = TypeParam{}.max_size() / 2;

    TypeParam v(count);
    EXPECT_EQ(v.size(), count);
}

TYPED_TEST(InplaceVectorTest, sizeConstructorOverflow)
{
    constexpr auto count = TypeParam{}.max_size() + 1;

    EXPECT_THROW((TypeParam(count)), std::bad_alloc);
}

TYPED_TEST(InplaceVectorTest, isIteratorConstructible)
{
    if constexpr (std::is_copy_constructible_v<typename TypeParam::value_type>) {
        const auto values = this->test_values();

        TypeParam v(values.begin(), values.end());
        EXPECT_THAT(v, ElementsAreArray(values));
    }
}

TYPED_TEST(InplaceVectorTest, isRangeConstructible)
{
    if constexpr (std::is_copy_constructible_v<typename TypeParam::value_type>) {
        const auto values = this->test_values();

        TypeParam v(std::from_range, values);
        EXPECT_THAT(v, ElementsAreArray(values));
    }
}

TYPED_TEST(InplaceVectorTest, isCopyConstructible)
{
    if constexpr (std::is_copy_constructible_v<typename TypeParam::value_type>) {
        const auto values = this->test_values();

        TypeParam v1(std::from_range, values);
        TypeParam v2(v1);
        EXPECT_THAT(v2, ElementsAreArray(values));
    }
}

TYPED_TEST(InplaceVectorTest, isMoveConstructible)
{
    TypeParam v1;
    for (auto& element : this->test_values()) {
        v1.emplace_back(std::move(element));
    }
    TypeParam v2(std::move(v1));
    auto values = this->test_values();
    EXPECT_TRUE(std::equal(v2.begin(), v2.end(), values.begin(), values.end()));
}

TYPED_TEST(InplaceVectorTest, isInitializerListConstructible)
{
    using value_type = typename TypeParam::value_type;
    if constexpr (std::is_copy_constructible_v<value_type>) {
        if (TypeParam{}.max_size() >= 3) {
            TypeParam v{value_type{}, value_type{}, value_type{}};
            EXPECT_EQ(v.size(), 3);
        }
    }
}