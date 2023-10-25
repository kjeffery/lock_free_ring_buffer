# Ring Buffer

The RingBuffer class is a thread-safe concurrent queue that supports multiple writers (producers) and multiple readers (consumers). The RingBuffer also has optimizations for single writers, single readers, and single writers that do batch insertion.

## Class Template Parameters

    template <typename T, std::size_t log_n_elements, typename Traits>
    class RingBuffer;

`T` is the contained type. `T` does not have to be default constructable, but it does require either a move constructor or a copy constructor, and a move assignment operator or a copy assignment operator.

`log_n_elements` is the base 2 logarithm for the capacity of the RingBuffer. For example, a value of 10 will create a RingBuffer with a capacity of 2<sup>10</sup> = 1024 elements.

`Traits` is a struct that must have the following static members:
- A `ProducerTraits` enum named `producer_traits`
- A `ConsumerTraits` enum named `consumer_traits`
- A `LockTraits` enum named `lock_traits`

`ProducerTraits` can take on the following values:
- `multiple_producers`: to allow for multiple threads to write to the RingBuffer.
- `single_producer`: to optimize when there is only a single thread writing to the RingBuffer.

Setting `single_producer` and then writing with multiple threads leads to undefined behavior.

`ConsumerTraits` can take on the following values:
- `multiple_consumers`: to allow for multiple threads to read from the RingBuffer.
- `single_consumer`: to optimize when there is only a single thread reading from the RingBuffer.

Setting `single_consumer` and then reading with multiple threads leads to undefined behavior.

`LockTraits` can take on the following values:
- `auto_detect`: which allows the RingBuffer to automatically choose the locking strategy.
- `spin_lock`: use a spin-lock in certain sections of the queue. If the contained type can throw exceptions, using this locking method leads to undefined behavior.
- `lock_free`: use a lock-free algorithm.

## Methods

- `RingBuffer()`: creates an empty queue.
- `LockTraits get_lock_traits()` (static): returns an enum of the lock traits.
- `bool empty() const` (thread-safe): returns `true` if the queue is empty.
- `bool full() const` (thread-safe): returns `true` if the queue is full.
- `index_t size() const` (thread-safe): returns the number of elements in the queue.
- `index_t capacity()` (static): returns the capacity of the queue.
- `bool try_push(const T& t)` (thread-safe): Attempts to copy into the queue. Returns `false` if the queue is full.
- `bool try_push(T&& t)` (thread-safe): Attempts to move into the queue. Returns `false` if the queue is full.
- `void push(const T& t)` (thread-safe): Copies into the queue, waiting until there is room.
- `void push(T&& t)` (thread-safe): Moves into the queue, waiting until there is room.
- `bool try_push_batch(Iterator first, Iterator last)` (thread-safe for one producer): Attempts to insert a range of values into the queue. If using a move iterator, it will attempt to move.
- `void push_batch(Iterator first, Iterator last)` (thread-safe for one producer): Inserts a range of values into the queue. If using a move iterator, it will attempt to move. If using random access iterators, the range may be larger than the queue capacity.
- `void emplace(Args&&... args)` (thread-safe): Attempts to insert `T` as built in-place by forwarding `args`.
- `bool try_pop(T& ret)` (thread-safe): Attempts to place the next item into `ret`. Returns `false` if the queue is empty.
- `T pop()` (thread-safe): Returns the next item in the queue, waiting until there is data.
- `void pop(T& ret)` (thread-safe): Returns the next item in the queue in `ret`, waiting until there is data.

### Exception Safety

The following do not throw exceptions:
- `get_lock_traits()`
- `empty()`
- `full()`
- `size()`
- `capacity()`

The following meet the strong exception guarantee (exceptions may be thrown, but the (apparent) state of the container is unchanged) when using the lock-free mode. Exceptions are propagated from the contained type's constructors or assignment operators. Exceptions in the spin-lock mode lead to undefined behavior.
- `try_push(const T& t)`
- `try_push(T&& t)`
- `push(const T& t)`
- `push(T&& t)`
- `emplace(Args&&... args)`

The following meet the basic exception guarantee (exceptions may be thrown, leaving the container in a consistent state, but the popped data is lost) owing to the return semantics of the container.
- `try_pop(T& ret)`
- `pop()`

## Efficiency Notes

Ideally, the contained type has a `noexcept` move constructor and a `noexcept` move assignment operator. This will lead to better performance.

If the type does have `noexcept` specifications, the RingBuffer will use a spin-lock by default. If the container capacity is large, you may get better performance explicitly setting lock-free traits.

The default multi-producer/multi-consumer traits will work in all cases, but if there is one producer, one consumer, or both one producer and one consumer, you will get a slight performance improvement by setting the traits appropriately.