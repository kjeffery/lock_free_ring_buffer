#include "RingBuffer.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#if defined(_MSC_FULL_VER)
#include <intrin.h>
std::uint16_t next_pow2(std::uint16_t x) noexcept
{
    return (x == 1) ? 1u : 1u << (16u - __lzcnt16(x - 1u));
}

std::uint32_t next_pow2(std::uint32_t x) noexcept
{
    return (x == 1) ? 1u : 1u << (32u - __lzcnt(x - 1u));
}

// Windows doesn't recognize this function if not compiled in 64-bit mode
//std::uint64_t next_pow2(std::uint64_t x) noexcept
//{
    //return (x == 1) ? 1u : 1u << (64u - __lzcnt64(x - 1u));
//}
#else
std::uint16_t next_pow2(std::uint16_t x) noexcept
{
    return (x == 1) ? 1u : 1u << (32u - __builtin_clz(x - 1u));
}

std::uint32_t next_pow2(std::uint32_t x) noexcept
{
    return (x == 1) ? 1u : 1u << (32u - __builtin_clz(x - 1u));
}

std::uint64_t next_pow2(std::uint64_t x) noexcept
{
    return (x == 1) ? 1u : 1u << (64u - __builtin_clzl(x - 1u));
}
#endif

std::uint64_t morton_encode(std::uint32_t x, std::uint32_t y) noexcept
{
    std::uint64_t a(x); 
    a = (a | (a << 16ull)) & 0x0000FFFF0000FFFFull;
    a = (a | (a <<  8ull)) & 0x00FF00FF00FF00FFull;
    a = (a | (a <<  4ull)) & 0x0F0F0F0F0F0F0F0Full;
    a = (a | (a <<  2ull)) & 0x3333333333333333ull;
    a = (a | (a <<  1ull)) & 0x5555555555555555ull;

    std::uint64_t b(y); 
    b = (b | (b << 16ull)) & 0x0000FFFF0000FFFFull;
    b = (b | (b <<  8ull)) & 0x00FF00FF00FF00FFull;
    b = (b | (b <<  4ull)) & 0x0F0F0F0F0F0F0F0Full;
    b = (b | (b <<  2ull)) & 0x3333333333333333ull;
    b = (b | (b <<  1ull)) & 0x5555555555555555ull;

    return a | (b << 1ull);
}

// morton_1 - extract even bits
uint32_t morton_decode_1(std::uint64_t a) noexcept
{
    a = a                  & 0x5555555555555555ull;
    a = (a | (a >> 1ull))  & 0x3333333333333333ull;
    a = (a | (a >> 2ull))  & 0x0F0F0F0F0F0F0F0Full;
    a = (a | (a >> 4ull))  & 0x00FF00FF00FF00FFull;
    a = (a | (a >> 8ull))  & 0x0000FFFF0000FFFFull;
    a = (a | (a >> 16ull)) & 0x00000000FFFFFFFFull;
    return static_cast<uint32_t>(a);
}

void morton_decode(std::uint64_t d, std::uint32_t& x, std::uint32_t& y) noexcept
{
    x = morton_decode_1(d);
    y = morton_decode_1(d >> 1ull);
}

std::uint32_t morton_encode(std::uint16_t x, std::uint16_t y) noexcept
{
    std::uint32_t a(x); 
    a = (a | (a << 8ul)) & 0x00FF00FFul;
    a = (a | (a << 4ul)) & 0x0F0F0F0Ful;
    a = (a | (a << 2ul)) & 0x33333333ul;
    a = (a | (a << 1ul)) & 0x55555555ul;

    std::uint32_t b(y); 
    b = (b | (b << 8ul)) & 0x00FF00FFul;
    b = (b | (b << 4ul)) & 0x0F0F0F0Ful;
    b = (b | (b << 2ul)) & 0x33333333ul;
    b = (b | (b << 1ul)) & 0x55555555ul;

    return a | (b << 1ul);
}

// morton_1 - extract even bits
uint16_t morton_decode_1(std::uint32_t a) noexcept
{
    a = a                 & 0x55555555ul;
    a = (a | (a >> 1ul))  & 0x33333333ul;
    a = (a | (a >> 2ul))  & 0x0F0F0F0Ful;
    a = (a | (a >> 4ul))  & 0x00FF00FFul;
    a = (a | (a >> 8ul))  & 0x0000FFFFul;
    return static_cast<uint16_t>(a);
}

std::uint16_t morton_decode_x(std::uint32_t d) noexcept
{
    return morton_decode_1(d);
}

std::uint16_t morton_decode_y(std::uint32_t d) noexcept
{
    return morton_decode_1(d >> 1ul);
}

void morton_decode(std::uint32_t d, std::uint16_t& x, std::uint16_t& y) noexcept
{
    x = morton_decode_x(d);
    y = morton_decode_y(d);
}

constexpr std::uint16_t k_tile_size = 8;

class Tile
{
public:
    explicit Tile(std::uint32_t n) noexcept
    : m_corner(n)
    {
    }

    std::uint16_t x() const noexcept
    {
        return morton_decode_x(m_corner) * k_tile_size;
    }

    std::uint16_t y() const noexcept
    {
        return morton_decode_y(m_corner) * k_tile_size;
    }

private:
    std::uint32_t m_corner;
};

struct RenderTile
{
    RenderTile()
    : m_tile(0)
    , m_sample_first(0)
    , m_sample_last(0)
    {
    }

    RenderTile(Tile tile, std::uint16_t first, std::uint16_t last) noexcept
    : m_tile(tile)
    , m_sample_first(first)
    , m_sample_last(last)
    {
    }

    explicit operator bool() const noexcept
    {
        return m_sample_first < m_sample_last;
    }

    Tile m_tile;
    std::uint16_t m_sample_first;
    std::uint16_t m_sample_last;
};

struct Pixel
{
    constexpr Pixel(std::uint16_t ix, std::uint16_t iy) noexcept
    : x(ix)
    , y(iy)
    {
    }

    std::uint16_t x;
    std::uint16_t y;
};

struct RGB
{
    constexpr RGB() noexcept
    : r(0)
    , g(0)
    , b(0)
    {
    }

    constexpr RGB(float ir, float ig, float ib) noexcept
    : r(ir)
    , g(ig)
    , b(ib)
    {
    }

    float r;
    float g;
    float b;
};

struct PixelColor
{
    constexpr PixelColor() noexcept
    : m_L(0.0f, 0.0f, 0.0f)
    , m_pixel(-1, -1)
    , m_x_offset(0.0f)
    , m_y_offset(0.0f)
    {
    }

    constexpr PixelColor(RGB L, std::uint16_t x, std::uint16_t y, float x_offset, float y_offset) noexcept
    : m_L(L)
    , m_pixel(x, y)
    , m_x_offset(x_offset)
    , m_y_offset(y_offset)
    {
    }

    RGB m_L;
    Pixel m_pixel;

    // We store the offset separately for floating-point accuracy. This may be used for filtering.
    float m_x_offset;
    float m_y_offset;
};

using TileBuffer     = RingBufferSingleProducer<RenderTile, 8>;
using RadianceBuffer = RingBufferSingleConsumer<PixelColor, 8>;
using Image          = std::unique_ptr<RGB[]>;

template <typename T>
T square(const T& x) noexcept(noexcept(x*x))
{
    return x*x;
}

// Wrap buffer in struct with this value?
std::atomic<bool> cpu_done(false);

RGB color(float x, float y, std::uint16_t nx, std::uint16_t ny)
{
    constexpr float k_magic_number = 3.79f;
    x = x/nx * k_magic_number;
    y = y/ny * k_magic_number;
    const float s = 0.5f * (1.0f + std::sin(square(x) * square(y)));
    return RGB{ s * 1.0f, s * 0.5f, s * 0.25f };
}

struct Renderer
{
    Renderer()
    : m_done(false)
    {
    }

    void operator()(TileBuffer& input_buffer, RadianceBuffer& output_buffer, std::uint16_t max_x, std::uint16_t max_y)
    {
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_real_distribution<float> dist;

        auto canon_u = [&rng, &dist] {
            float u;
            do {
                u = dist(rng);
            } while (u >= 1.0f);
            assert(u < 1.0f);
            return u;
        };

        while (true) {
            RenderTile tile;
            if (!input_buffer.try_pop(tile)) {
                // If we're done, we still want to drain the queue
                if (cpu_done && !input_buffer.try_pop(tile)) {
                    m_done = true;
                    return;
                }
                continue;
            }

            const std::uint16_t tile_lower_x = tile.m_tile.x();
            const std::uint16_t tile_lower_y = tile.m_tile.y();

            for (std::uint16_t j = 0; j < k_tile_size; ++j) {
                for (std::uint16_t i = 0; i < k_tile_size; ++i) {
                    const std::uint16_t ix = tile_lower_x + i;
                    const std::uint16_t iy = tile_lower_y + j;
                    if (ix >= max_x || iy >= max_y) {
                        continue;
                    }

                    for (auto s = tile.m_sample_first; s != tile.m_sample_last; ++s) {
                        const float x_offset = canon_u();
                        const float y_offset = canon_u();
                        const RGB c = color(ix + x_offset, iy + y_offset, max_x, max_y);
                        output_buffer.emplace(c, ix, iy, x_offset, y_offset);
                    }
                }
            }
        }
    }

    std::atomic_bool m_done;
};

float rgb_to_srgb(float u) noexcept
{
    if (u <= 0.0031308f) {
        return 12.92f * u;
    } else {
        return 1.055f * std::pow(u, 1.0f/2.4f) - 0.055f;
    }
}

RGB rgb_to_srgb(const RGB& c) noexcept
{
    return RGB{rgb_to_srgb(c.r), rgb_to_srgb(c.g), rgb_to_srgb(c.b)};
}

void output_ppm(std::ostream& outs, const RGB* fb, int nx, int ny)
{
    outs << "P3\n" << nx << ' ' << ny << "\n255\n";
    for (int j = ny-1; j >= 0; --j) {
        for (int i = 0; i < nx; ++i) {
            const std::size_t pixel_index = j*nx + i;
            //const RGB c = rgb_to_srgb(fb[pixel_index]);
            const RGB& c = fb[pixel_index];
            const int ir = static_cast<int>(255.99f*c.r);
            const int ig = static_cast<int>(255.99f*c.g);
            const int ib = static_cast<int>(255.99f*c.b);
            outs << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
}

std::uint32_t drain_radiance_buffer(RadianceBuffer& radiance_buffer,
                                    Image& image,
                                    std::uint16_t nx,
                                    std::uint16_t /*ny*/,
                                    std::uint32_t nvalues = std::numeric_limits<std::uint32_t>::max())
{
    nvalues = std::min(radiance_buffer.size(), nvalues);
    for (std::uint32_t i = 0; i < nvalues; ++i) {
        PixelColor c;
        if (!radiance_buffer.try_pop(c)) {
            return i;
        }
        image[c.m_pixel.y * nx + c.m_pixel.x].r += c.m_L.r;
        image[c.m_pixel.y * nx + c.m_pixel.x].g += c.m_L.g;
        image[c.m_pixel.y * nx + c.m_pixel.x].b += c.m_L.b;
    }
    return nvalues;
}

unsigned get_number_cpu_worker_threads()
{
    const unsigned hc = std::thread::hardware_concurrency();
    // We need at least one worker thread, but default to one less than hardware support, so that we can leave the main
    // thread to do its thing.
    const unsigned n_cpu_threads = std::max(1u, (hc == 0) ? 1 : std::thread::hardware_concurrency() - 1);
    //const unsigned n_cpu_threads = 1;
    return n_cpu_threads;
}

class GenerateRenderTiles
{
public:
    GenerateRenderTiles(std::uint16_t nx, std::uint16_t ny, std::uint16_t nsamples) noexcept
    : mNSamples(nsamples)
    , mFirst(0)
    , mLast(1)
    , mCurrentTile(0)
    {
        const std::uint16_t ntiles_x = nx/k_tile_size + 1;
        const std::uint16_t ntiles_y = ny/k_tile_size + 1;

        // We have two options:
        // 1. Go through x and y in scanline order, sort at the end (O(n log n)), and find discontinuities
        // 2. Round to larger power of two, and go through numbers sequentually, decode, and note discontinuties and cull out values

        // We want a perfectly square power of two number of tiles that covers all of the actual tiles we have so that we
        // can just iterate over the Morton values.

        const std::uint16_t p2_tiles = next_pow2(square(std::max(ntiles_x, ntiles_y)));
        for (std::uint16_t i = 0; i < p2_tiles; ++i) {
            const Tile tile(i);
            if (tile.x() >= nx || tile.y() >= ny) {
                continue;
            }
            mTiles.push_back(tile);
        }
    }

    RenderTile operator()()
    {
        if (mCurrentTile == mTiles.size()) {
            mFirst = mLast;
            mLast *= 2;
            mLast = std::min(mNSamples, mLast);
            mCurrentTile = 0;
        }

        Tile& tile = mTiles[mCurrentTile++];
        return RenderTile{tile, mFirst, mLast};
    }

    bool done() const noexcept
    {
        return mFirst == mLast;
    }

private:
    std::uint16_t mNSamples;
    std::uint16_t mFirst;
    std::uint16_t mLast;
    std::size_t mCurrentTile;
    std::vector<Tile> mTiles;
};

constexpr std::uint16_t nsamples = 128;

// TODO: a proper image class with x and y
void render_loop(Image& image, std::uint16_t nx, std::uint16_t ny, TileBuffer& tile_buffer, RadianceBuffer& radiance_buffer)
{
    GenerateRenderTiles tile_generator(nx, ny, nsamples);

    // Save up the tiles we've generated for batch insertion into the queue
    std::vector<RenderTile> render_tiles_accumulator;
    render_tiles_accumulator.reserve(tile_buffer.capacity());

    constexpr int n_tile_queues = 1; // This will increase with however many GPU queues...
    constexpr int n_radiance_queues = 1; // This will increase with however many GPU queues...
    while (true) {
        for (int i = 0; i < n_radiance_queues; ++i) {
            drain_radiance_buffer(radiance_buffer, image, nx, ny);
        }
        for (int i = 0; i < n_tile_queues; ++i) {
            render_tiles_accumulator.clear();
            const auto size = tile_buffer.size();
            for (int j = 0; j < tile_buffer.capacity() - size; ++j) {
                auto t = tile_generator();
                if (t) {
                    render_tiles_accumulator.push_back(std::move(t));
                }
                if (tile_generator.done()) {
                    break;
                }
            }
            const auto ntiles = render_tiles_accumulator.size();
            if (ntiles == 1) {
                tile_buffer.push(render_tiles_accumulator.front());
            } else if (ntiles > 1) {
                tile_buffer.push_batch(render_tiles_accumulator.cbegin(), render_tiles_accumulator.cend());
            }

            if (tile_generator.done()) {
                return;
            }
        }
    }
}

#if 0
class TileQueueAbstract
{
public:
    template <typename T>
    explicit TileQueueAbstract(T queue)
    {

    }

private:
    struct QueueBase
    {
        virtual void try_push_batch() = 0; // Template? Fux.
    };

    template <typename QType>
    struct Queue : public QueueBase
    {
        //explicit Queue(QType)

    }
};

void render_loop(Image& image, std::uint16_t nx, std::uint16_t ny, TileBuffer& tile_buffer, RadianceBuffer& radiance_buffer)
{
    // Setup queues (in caller?)
    std::vector<TileQueueAbstract> queues;

    // Go through each queue and calculate available space (capacity - size)
    // Warn if (aside from first attempt) buffer completely empty. We're not using our full throughput.
    // s = Sum available space
    // Ask tile generator for s tiles
    // If tile generator generates s tiles:
    // distribute exactly requested number
    // Else
    // distribute based on weighted average

}
#endif

void render(std::uint16_t nx, std::uint16_t ny)
{
    Image image(new RGB[nx * ny]);

    const unsigned n_cpu_threads = get_number_cpu_worker_threads();
    //const unsigned n_cpu_threads = 1;
    std::cout << "Rendering with " << n_cpu_threads << " cpu threads\n";

    TileBuffer     tile_buffer;
    RadianceBuffer radiance_buffer;

    std::vector<Renderer> cpu_renderers(n_cpu_threads);

    std::vector<std::thread> cpu_threads;
    cpu_threads.reserve(n_cpu_threads);

    for (unsigned i = 0; i < n_cpu_threads; ++i) {
        cpu_threads.emplace_back(std::ref(cpu_renderers.at(i)), std::ref(tile_buffer), std::ref(radiance_buffer), nx, ny);
    }

    render_loop(image, nx, ny, tile_buffer, radiance_buffer);

    // While we still have active tiles, keep draining the radiance buffer to make sure there's room.
    while (!tile_buffer.empty()) {
        drain_radiance_buffer(radiance_buffer, image, nx, ny);
    }

    std::cout << "Producer done" << std::endl;
    cpu_done = true;

    for (unsigned i = 0; i < n_cpu_threads; ++i) {
        while (!cpu_renderers[i].m_done) {
            drain_radiance_buffer(radiance_buffer, image, nx, ny);
        }
    }

    while (drain_radiance_buffer(radiance_buffer, image, nx, ny)) ;
    for (auto& t : cpu_threads) {
        t.join();
        std::cout << "Render thread done" << std::endl;
    }
    // We don't need this...
    while (drain_radiance_buffer(radiance_buffer, image, nx, ny)) ;

    // TODO: We want a different sample count for each pixel. A separate buffer?
    for (std::uint16_t y = 0; y < ny; ++y) {
        for (std::uint16_t x = 0; x < nx; ++x) {
            image[y * nx + x].r /= nsamples;
            image[y * nx + x].g /= nsamples;
            image[y * nx + x].b /= nsamples;
        }
    }

    std::filesystem::path output_file("c:/Users/Keith/projects/RingBuffer/image.ppm");
    std::ofstream outs(output_file);
    if (!outs) {
        std::cerr << "Unable to open output file '" << output_file << "'\n";
    }
    output_ppm(outs, image.get(), nx, ny);

    // Allocate two producer queues: GPU and CPU
    // Allocate two consumer queues: GPU and CPU
    // Fill each producer queue half way
    // Keep track of number of elements in each producer queue
    // Pull results from consumer queue
    // Allocate enough producer elements to fill halfway
}
