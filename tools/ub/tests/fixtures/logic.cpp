// Кириллица exercises Clang's byte offsets.
namespace fixture {
int choose(int x) {
    if (x > 0 && x < 10) return x + 1;
    return 0;
}
int duplicate_a(int x) {
    int total = x;
    total += 1;
    total *= 2;
    total -= 3;
    return total;
}
int duplicate_b(int x) {
    int total = x;
    total += 1;
    total *= 2;
    total -= 3;
    return total;
}
int not_executed(int x) {
    if (x) return 1;
    return 0;
}
template<class T> T identity(T x) { return x; }
int instantiate() { return identity(1) + int(identity(2.0)); }
int with_lambda(int x) {
    auto test = [](int n) { if (n) return 1; return 0; };
    return test(x);
}
struct Owner {
    Owner(int n) try : value(n) { if (n < 0) value = 0; } catch (...) { throw; }
    ~Owner() { if (value) value = 0; }
    int value;
};
}
int main() {
    fixture::Owner owner(1);
    return fixture::choose(2) != 3 || fixture::duplicate_a(3) != 5 ||
           fixture::instantiate() != 3 || fixture::with_lambda(1) != 1;
}
