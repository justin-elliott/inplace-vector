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

#include "detail/storage.hpp"

namespace jell::detail::inplace_vector {

/// A class modelling an exception-safe attic into which elements can be moved during vector modification.
template <typename T, std::size_t N>
class attic
{
private:
    using storage_type = detail::inplace_vector::storage<T, N>;

public:
    using size_type = storage_type::size_type;

    /// Destructively move-construct elements in the range [save_pos..storage.size()) into the attic,
    /// [attic_end - storage.size() + save_pos..attic_end).
    /// @param storage The storage in which to move elements.
    /// @param save_pos The position from which to move elements.
    /// @param attic_end The end position of the attic, into which to save elements.
    template <std::random_access_iterator Iterator>
    constexpr attic(storage_type& storage, Iterator save_pos, size_type attic_end)
        : storage_{storage}
        , begin_{attic_end}
        , end_{attic_end}
    {
        // Note that operator->() explicitly allows dereferencing at the end().
        const auto save_index = static_cast<size_type>(save_pos.operator->() - storage_.data());
    
        if (begin_ == storage_.size())
        {
            begin_ = save_index;
            storage_.size(begin_);
        } else {
            for (; storage_.size() != save_index; --begin_) {
                const auto last_index = storage_.size() - 1;
                storage_.construct_at(begin_ - 1, std::move(storage_.data()[last_index]));
                storage_.destroy_at(last_index);
                storage_.size(last_index);
            }
        }
    }

    /// Destroy any remaining entries in the attic (typically only during an exception).
    constexpr ~attic()
    {
        storage_.destroy(begin_, end_);
    }

    /// Retrieve all elements from the attic, destructively move-constructing them if they are not already in their
    /// required location, and adjust the storage.size().
    constexpr void retrieve()
    {
        if (storage_.size() == begin_) {
            begin_ = end_;
            storage_.size(end_);
        } else {
            for (; begin_ != end_; ++begin_) {
                storage_.construct_at(storage_.size(), std::move(storage_.data()[begin_]));
                storage_.destroy_at(begin_);
                storage_.size(storage_.size() + 1);
            }
        }
    }

    /// Check that the position is not within the bounds of the attic or above, throwing bad_alloc if the check fails.
    /// @param pos The position to check.
    constexpr void capacity_check(size_type pos) const
    {
        if (pos >= begin_)
        {
            throw std::bad_alloc{};
        }
    }

private:
    storage_type& storage_;
    size_type begin_;
    size_type end_;
};

} // namespace jell::detail::inplace_vector
