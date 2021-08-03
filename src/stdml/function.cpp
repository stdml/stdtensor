#include <experimental/iterator>
#include <sstream>

#include <stdml/bits/function.hpp>
#include <stdml/bits/tensor/io.hpp>

namespace stdml
{
std::string Function::call_info(const Y &ys, const X &xs)
{
    std::stringstream ss;
    std::transform(ys.begin(), ys.end(),
                   std::experimental::make_ostream_joiner(ss, ", "),
                   [](auto &t) { return stdml::info(t); });
    ss << " <- ";
    std::transform(xs.begin(), xs.end(),
                   std::experimental::make_ostream_joiner(ss, ", "),
                   [](auto &t) { return stdml::info(t); });
    return ss.str();
}

Function::Kind Function::operator()(const std::vector<Kind> &ks) const
{
    std::vector<DType> dtypes;
    std::vector<Shape> shapes;
    std::transform(ks.begin(), ks.end(), std::back_inserter(dtypes),
                   [](auto &p) { return p.first; });
    std::transform(ks.begin(), ks.end(), std::back_inserter(shapes),
                   [](auto &p) { return p.second; });
    return Kind((*this)(dtypes), (*this)(shapes));
}
}  // namespace stdml
