use crate::field::F128;

pub(super) const TWIDDLE_COUNT: usize = (1 << 8) - 1;
pub(super) const TAB8_PER_TWIDDLE: usize = 256;
pub(super) const TAB8_LEN: usize = TWIDDLE_COUNT * TAB8_PER_TWIDDLE;

const TWIDDLES: [F128; TWIDDLE_COUNT] = [
    F128 {
        lo: 0x0000000000000000,
        hi: 0x0000000000000000,
    },
    F128 {
        lo: 0x0000000000000000,
        hi: 0x0000000000000000,
    },
    F128 {
        lo: 0x282fb37510c7a273,
        hi: 0x1d6810db75596418,
    },
    F128 {
        lo: 0x0000000000000000,
        hi: 0x0000000000000000,
    },
    F128 {
        lo: 0xc24e3d64cfc3486d,
        hi: 0x9601404044740119,
    },
    F128 {
        lo: 0xd47a03db57d43566,
        hi: 0xaa83e5cb2e43659c,
    },
    F128 {
        lo: 0x16343ebf98177d0b,
        hi: 0x3c82a58b6a376485,
    },
    F128 {
        lo: 0x0000000000000000,
        hi: 0x0000000000000000,
    },
    F128 {
        lo: 0x5188afa3badb31ea,
        hi: 0x10000092106087df,
    },
    F128 {
        lo: 0x7f79be777eafb70b,
        hi: 0x35bf8abc6c2916e4,
    },
    F128 {
        lo: 0x2ef111d4c47486e1,
        hi: 0x25bf8a2e7c49913b,
    },
    F128 {
        lo: 0x2d8de3b47fe05f02,
        hi: 0xba337bc619cc4756,
    },
    F128 {
        lo: 0x7c054c17c53b6ee8,
        hi: 0xaa337b5409acc089,
    },
    F128 {
        lo: 0x52f45dc3014fe809,
        hi: 0x8f8cf17a75e551b2,
    },
    F128 {
        lo: 0x037cf260bb94d9e3,
        hi: 0x9f8cf1e86585d66d,
    },
    F128 {
        lo: 0x0000000000000000,
        hi: 0x0000000000000000,
    },
    F128 {
        lo: 0x400749bad3a3de9a,
        hi: 0x0001008301718ebe,
    },
    F128 {
        lo: 0x7547113c646373da,
        hi: 0x01ce97207bad2213,
    },
    F128 {
        lo: 0x35405886b7c0ad40,
        hi: 0x01cf97a37adcacad,
    },
    F128 {
        lo: 0x37d978f882f7ee15,
        hi: 0xa4d55a2a347958f6,
    },
    F128 {
        lo: 0x77de31425154308f,
        hi: 0xa4d45aa93508d648,
    },
    F128 {
        lo: 0x429e69c4e6949dcf,
        hi: 0xa51bcd0a4fd47ae5,
    },
    F128 {
        lo: 0x0299207e35374355,
        hi: 0xa51acd894ea5f45b,
    },
    F128 {
        lo: 0x505f19cb6c6ee5cf,
        hi: 0xca7b4433ed3a67bc,
    },
    F128 {
        lo: 0x10585071bfcd3b55,
        hi: 0xca7a44b0ec4be902,
    },
    F128 {
        lo: 0x251808f7080d9615,
        hi: 0xcbb5d313969745af,
    },
    F128 {
        lo: 0x651f414ddbae488f,
        hi: 0xcbb4d39097e6cb11,
    },
    F128 {
        lo: 0x67866133ee990bda,
        hi: 0x6eae1e19d9433f4a,
    },
    F128 {
        lo: 0x278128893d3ad540,
        hi: 0x6eaf1e9ad832b1f4,
    },
    F128 {
        lo: 0x12c1700f8afa7800,
        hi: 0x6f608939a2ee1d59,
    },
    F128 {
        lo: 0x52c639b55959a69a,
        hi: 0x6f6189baa39f93e7,
    },
    F128 {
        lo: 0x0000000000000000,
        hi: 0x0000000000000000,
    },
    F128 {
        lo: 0x01021caf83f68eda,
        hi: 0x000100870021caea,
    },
    F128 {
        lo: 0x19e5cc07628fa323,
        hi: 0x017d0dd476821087,
    },
    F128 {
        lo: 0x18e7d0a8e1792df9,
        hi: 0x017c0d5376a3da6d,
    },
    F128 {
        lo: 0x7ef2cd0fa97e54c7,
        hi: 0xc209cbdecd95b097,
    },
    F128 {
        lo: 0x7ff0d1a02a88da1d,
        hi: 0xc208cb59cdb47a7d,
    },
    F128 {
        lo: 0x67170108cbf1f7e4,
        hi: 0xc374c60abb17a010,
    },
    F128 {
        lo: 0x66151da74807793e,
        hi: 0xc375c68dbb366afa,
    },
    F128 {
        lo: 0xf5e7600704135fb1,
        hi: 0x581bc8592a8d45bb,
    },
    F128 {
        lo: 0xf4e57ca887e5d16b,
        hi: 0x581ac8de2aac8f51,
    },
    F128 {
        lo: 0xec02ac00669cfc92,
        hi: 0x5966c58d5c0f553c,
    },
    F128 {
        lo: 0xed00b0afe56a7248,
        hi: 0x5967c50a5c2e9fd6,
    },
    F128 {
        lo: 0x8b15ad08ad6d0b76,
        hi: 0x9a120387e718f52c,
    },
    F128 {
        lo: 0x8a17b1a72e9b85ac,
        hi: 0x9a130300e7393fc6,
    },
    F128 {
        lo: 0x92f0610fcfe2a855,
        hi: 0x9b6f0e53919ae5ab,
    },
    F128 {
        lo: 0x93f27da04c14268f,
        hi: 0x9b6e0ed491bb2f41,
    },
    F128 {
        lo: 0x574e655224bb0402,
        hi: 0xb39529e28d9ec7f7,
    },
    F128 {
        lo: 0x564c79fda74d8ad8,
        hi: 0xb39429658dbf0d1d,
    },
    F128 {
        lo: 0x4eaba9554634a721,
        hi: 0xb2e82436fb1cd770,
    },
    F128 {
        lo: 0x4fa9b5fac5c229fb,
        hi: 0xb2e924b1fb3d1d9a,
    },
    F128 {
        lo: 0x29bca85d8dc550c5,
        hi: 0x719ce23c400b7760,
    },
    F128 {
        lo: 0x28beb4f20e33de1f,
        hi: 0x719de2bb402abd8a,
    },
    F128 {
        lo: 0x3059645aef4af3e6,
        hi: 0x70e1efe8368967e7,
    },
    F128 {
        lo: 0x315b78f56cbc7d3c,
        hi: 0x70e0ef6f36a8ad0d,
    },
    F128 {
        lo: 0xa2a9055520a85bb3,
        hi: 0xeb8ee1bba713824c,
    },
    F128 {
        lo: 0xa3ab19faa35ed569,
        hi: 0xeb8fe13ca73248a6,
    },
    F128 {
        lo: 0xbb4cc9524227f890,
        hi: 0xeaf3ec6fd19192cb,
    },
    F128 {
        lo: 0xba4ed5fdc1d1764a,
        hi: 0xeaf2ece8d1b05821,
    },
    F128 {
        lo: 0xdc5bc85a89d60f74,
        hi: 0x29872a656a8632db,
    },
    F128 {
        lo: 0xdd59d4f50a2081ae,
        hi: 0x29862ae26aa7f831,
    },
    F128 {
        lo: 0xc5be045deb59ac57,
        hi: 0x28fa27b11c04225c,
    },
    F128 {
        lo: 0xc4bc18f268af228d,
        hi: 0x28fb27361c25e8b6,
    },
    F128 {
        lo: 0x0000000000000000,
        hi: 0x0000000000000000,
    },
    F128 {
        lo: 0x0100000110014112,
        hi: 0x000100870021caea,
    },
    F128 {
        lo: 0x7394410be6bc79f6,
        hi: 0xdb9798126ac040aa,
    },
    F128 {
        lo: 0x7294410af6bd38e4,
        hi: 0xdb9698956ae18a40,
    },
    F128 {
        lo: 0x62f91d60d49c08aa,
        hi: 0xa512f7ecdf249f6e,
    },
    F128 {
        lo: 0x63f91d61c49d49b8,
        hi: 0xa513f76bdf055584,
    },
    F128 {
        lo: 0x116d5c6b3220715c,
        hi: 0x7e856ffeb5e4dfc4,
    },
    F128 {
        lo: 0x106d5c6a2221304e,
        hi: 0x7e846f79b5c5152e,
    },
    F128 {
        lo: 0xdb451089d6543416,
        hi: 0x16953e65b2704aae,
    },
    F128 {
        lo: 0xda451088c6557504,
        hi: 0x16943ee2b2518044,
    },
    F128 {
        lo: 0xa8d1518230e84de0,
        hi: 0xcd02a677d8b00a04,
    },
    F128 {
        lo: 0xa9d1518320e90cf2,
        hi: 0xcd03a6f0d891c0ee,
    },
    F128 {
        lo: 0xb9bc0de902c83cbc,
        hi: 0xb387c9896d54d5c0,
    },
    F128 {
        lo: 0xb8bc0de812c97dae,
        hi: 0xb386c90e6d751f2a,
    },
    F128 {
        lo: 0xca284ce2e474454a,
        hi: 0x6810519b0794956a,
    },
    F128 {
        lo: 0xcb284ce3f4750458,
        hi: 0x6811511c07b55f80,
    },
    F128 {
        lo: 0xa2ca21c32ec83225,
        hi: 0x6e725edec5adf339,
    },
    F128 {
        lo: 0xa3ca21c23ec97337,
        hi: 0x6e735e59c58c39d3,
    },
    F128 {
        lo: 0xd15e60c8c8744bd3,
        hi: 0xb5e5c6ccaf6db393,
    },
    F128 {
        lo: 0xd05e60c9d8750ac1,
        hi: 0xb5e4c64baf4c7979,
    },
    F128 {
        lo: 0xc0333ca3fa543a8f,
        hi: 0xcb60a9321a896c57,
    },
    F128 {
        lo: 0xc1333ca2ea557b9d,
        hi: 0xcb61a9b51aa8a6bd,
    },
    F128 {
        lo: 0xb3a77da81ce84379,
        hi: 0x10f7312070492cfd,
    },
    F128 {
        lo: 0xb2a77da90ce9026b,
        hi: 0x10f631a77068e617,
    },
    F128 {
        lo: 0x798f314af89c0633,
        hi: 0x78e760bb77ddb997,
    },
    F128 {
        lo: 0x788f314be89d4721,
        hi: 0x78e6603c77fc737d,
    },
    F128 {
        lo: 0x0a1b70411e207fc5,
        hi: 0xa370f8a91d1df93d,
    },
    F128 {
        lo: 0x0b1b70400e213ed7,
        hi: 0xa371f82e1d3c33d7,
    },
    F128 {
        lo: 0x1b762c2a2c000e99,
        hi: 0xddf59757a8f926f9,
    },
    F128 {
        lo: 0x1a762c2b3c014f8b,
        hi: 0xddf497d0a8d8ec13,
    },
    F128 {
        lo: 0x68e26d21cabc776f,
        hi: 0x06620f45c2396653,
    },
    F128 {
        lo: 0x69e26d20dabd367d,
        hi: 0x06630fc2c218acb9,
    },
    F128 {
        lo: 0x1b5406bf9f65dd6f,
        hi: 0x6aff807f9500bc87,
    },
    F128 {
        lo: 0x1a5406be8f649c7d,
        hi: 0x6afe80f89521766d,
    },
    F128 {
        lo: 0x68c047b479d9a499,
        hi: 0xb168186dffc0fc2d,
    },
    F128 {
        lo: 0x69c047b569d8e58b,
        hi: 0xb16918eaffe136c7,
    },
    F128 {
        lo: 0x79ad1bdf4bf9d5c5,
        hi: 0xcfed77934a2423e9,
    },
    F128 {
        lo: 0x78ad1bde5bf894d7,
        hi: 0xcfec77144a05e903,
    },
    F128 {
        lo: 0x0a395ad4ad45ac33,
        hi: 0x147aef8120e46343,
    },
    F128 {
        lo: 0x0b395ad5bd44ed21,
        hi: 0x147bef0620c5a9a9,
    },
    F128 {
        lo: 0xc01116364931e979,
        hi: 0x7c6abe1a2770f629,
    },
    F128 {
        lo: 0xc11116375930a86b,
        hi: 0x7c6bbe9d27513cc3,
    },
    F128 {
        lo: 0xb385573daf8d908f,
        hi: 0xa7fd26084db0b683,
    },
    F128 {
        lo: 0xb285573cbf8cd19d,
        hi: 0xa7fc268f4d917c69,
    },
    F128 {
        lo: 0xa2e80b569dade1d3,
        hi: 0xd97849f6f8546947,
    },
    F128 {
        lo: 0xa3e80b578daca0c1,
        hi: 0xd9794971f875a3ad,
    },
    F128 {
        lo: 0xd17c4a5d7b119825,
        hi: 0x02efd1e4929429ed,
    },
    F128 {
        lo: 0xd07c4a5c6b10d937,
        hi: 0x02eed16392b5e307,
    },
    F128 {
        lo: 0xb99e277cb1adef4a,
        hi: 0x048ddea150ad4fbe,
    },
    F128 {
        lo: 0xb89e277da1acae58,
        hi: 0x048cde26508c8554,
    },
    F128 {
        lo: 0xca0a6677571196bc,
        hi: 0xdf1a46b33a6d0f14,
    },
    F128 {
        lo: 0xcb0a66764710d7ae,
        hi: 0xdf1b46343a4cc5fe,
    },
    F128 {
        lo: 0xdb673a1c6531e7e0,
        hi: 0xa19f294d8f89d0d0,
    },
    F128 {
        lo: 0xda673a1d7530a6f2,
        hi: 0xa19e29ca8fa81a3a,
    },
    F128 {
        lo: 0xa8f37b17838d9e16,
        hi: 0x7a08b15fe549907a,
    },
    F128 {
        lo: 0xa9f37b16938cdf04,
        hi: 0x7a09b1d8e5685a90,
    },
    F128 {
        lo: 0x62db37f567f9db5c,
        hi: 0x1218e0c4e2dd0510,
    },
    F128 {
        lo: 0x63db37f477f89a4e,
        hi: 0x1219e043e2fccffa,
    },
    F128 {
        lo: 0x114f76fe8145a2aa,
        hi: 0xc98f78d6881d45ba,
    },
    F128 {
        lo: 0x104f76ff9144e3b8,
        hi: 0xc98e7851883c8f50,
    },
    F128 {
        lo: 0x00222a95b365d3f6,
        hi: 0xb70a17283df99a7e,
    },
    F128 {
        lo: 0x01222a94a36492e4,
        hi: 0xb70b17af3dd85094,
    },
    F128 {
        lo: 0x73b66b9e55d9aa00,
        hi: 0x6c9d8f3a5739dad4,
    },
    F128 {
        lo: 0x72b66b9f45d8eb12,
        hi: 0x6c9c8fbd5718103e,
    },
    F128 {
        lo: 0x0000000000000000,
        hi: 0x0000000000000000,
    },
    F128 {
        lo: 0x0100000110014194,
        hi: 0x0001008700000000,
    },
    F128 {
        lo: 0x840092c915fcf930,
        hi: 0x31320eed4124caaa,
    },
    F128 {
        lo: 0x850092c805fdb8a4,
        hi: 0x31330e6a4124caaa,
    },
    F128 {
        lo: 0x9b50d0b508f5e36f,
        hi: 0x4ba7b1b0ec86688e,
    },
    F128 {
        lo: 0x9a50d0b418f4a2fb,
        hi: 0x4ba6b137ec86688e,
    },
    F128 {
        lo: 0x1f50427c1d091a5f,
        hi: 0x7a95bf5dada2a224,
    },
    F128 {
        lo: 0x1e50427d0d085bcb,
        hi: 0x7a94bfdaada2a224,
    },
    F128 {
        lo: 0x30e2f240037586dd,
        hi: 0x4c13b29bdc2b2bbe,
    },
    F128 {
        lo: 0x31e2f2411374c749,
        hi: 0x4c12b21cdc2b2bbe,
    },
    F128 {
        lo: 0xb4e2608916897fed,
        hi: 0x7d21bc769d0fe114,
    },
    F128 {
        lo: 0xb5e2608806883e79,
        hi: 0x7d20bcf19d0fe114,
    },
    F128 {
        lo: 0xabb222f50b8065b2,
        hi: 0x07b4032b30ad4330,
    },
    F128 {
        lo: 0xaab222f41b812426,
        hi: 0x07b503ac30ad4330,
    },
    F128 {
        lo: 0x2fb2b03c1e7c9c82,
        hi: 0x36860dc67189899a,
    },
    F128 {
        lo: 0x2eb2b03d0e7ddd16,
        hi: 0x36870d417189899a,
    },
    F128 {
        lo: 0x417736dbded43959,
        hi: 0x4b2fe4d092481094,
    },
    F128 {
        lo: 0x407736daced578cd,
        hi: 0x4b2ee45792481094,
    },
    F128 {
        lo: 0xc577a412cb28c069,
        hi: 0x7a1dea3dd36cda3e,
    },
    F128 {
        lo: 0xc477a413db2981fd,
        hi: 0x7a1ceabad36cda3e,
    },
    F128 {
        lo: 0xda27e66ed621da36,
        hi: 0x008855607ece781a,
    },
    F128 {
        lo: 0xdb27e66fc6209ba2,
        hi: 0x008955e77ece781a,
    },
    F128 {
        lo: 0x5e2774a7c3dd2306,
        hi: 0x31ba5b8d3feab2b0,
    },
    F128 {
        lo: 0x5f2774a6d3dc6292,
        hi: 0x31bb5b0a3feab2b0,
    },
    F128 {
        lo: 0x7195c49bdda1bf84,
        hi: 0x073c564b4e633b2a,
    },
    F128 {
        lo: 0x7095c49acda0fe10,
        hi: 0x073d56cc4e633b2a,
    },
    F128 {
        lo: 0xf5955652c85d46b4,
        hi: 0x360e58a60f47f180,
    },
    F128 {
        lo: 0xf4955653d85c0720,
        hi: 0x360f58210f47f180,
    },
    F128 {
        lo: 0xeac5142ed5545ceb,
        hi: 0x4c9be7fba2e553a4,
    },
    F128 {
        lo: 0xebc5142fc5551d7f,
        hi: 0x4c9ae77ca2e553a4,
    },
    F128 {
        lo: 0x6ec586e7c0a8a5db,
        hi: 0x7da9e916e3c1990e,
    },
    F128 {
        lo: 0x6fc586e6d0a9e44f,
        hi: 0x7da8e991e3c1990e,
    },
    F128 {
        lo: 0x95db661c3bd9669c,
        hi: 0xaf9ef94e4edba772,
    },
    F128 {
        lo: 0x94db661d2bd82708,
        hi: 0xaf9ff9c94edba772,
    },
    F128 {
        lo: 0x11dbf4d52e259fac,
        hi: 0x9eacf7a30fff6dd8,
    },
    F128 {
        lo: 0x10dbf4d43e24de38,
        hi: 0x9eadf7240fff6dd8,
    },
    F128 {
        lo: 0x0e8bb6a9332c85f3,
        hi: 0xe43948fea25dcffc,
    },
    F128 {
        lo: 0x0f8bb6a8232dc467,
        hi: 0xe4384879a25dcffc,
    },
    F128 {
        lo: 0x8a8b246026d07cc3,
        hi: 0xd50b4613e3790556,
    },
    F128 {
        lo: 0x8b8b246136d13d57,
        hi: 0xd50a4694e3790556,
    },
    F128 {
        lo: 0xa539945c38ace041,
        hi: 0xe38d4bd592f08ccc,
    },
    F128 {
        lo: 0xa439945d28ada1d5,
        hi: 0xe38c4b5292f08ccc,
    },
    F128 {
        lo: 0x213906952d501971,
        hi: 0xd2bf4538d3d44666,
    },
    F128 {
        lo: 0x203906943d5158e5,
        hi: 0xd2be45bfd3d44666,
    },
    F128 {
        lo: 0x3e6944e93059032e,
        hi: 0xa82afa657e76e442,
    },
    F128 {
        lo: 0x3f6944e8205842ba,
        hi: 0xa82bfae27e76e442,
    },
    F128 {
        lo: 0xba69d62025a5fa1e,
        hi: 0x9918f4883f522ee8,
    },
    F128 {
        lo: 0xbb69d62135a4bb8a,
        hi: 0x9919f40f3f522ee8,
    },
    F128 {
        lo: 0xd4ac50c7e50d5fc5,
        hi: 0xe4b11d9edc93b7e6,
    },
    F128 {
        lo: 0xd5ac50c6f50c1e51,
        hi: 0xe4b01d19dc93b7e6,
    },
    F128 {
        lo: 0x50acc20ef0f1a6f5,
        hi: 0xd58313739db77d4c,
    },
    F128 {
        lo: 0x51acc20fe0f0e761,
        hi: 0xd58213f49db77d4c,
    },
    F128 {
        lo: 0x4ffc8072edf8bcaa,
        hi: 0xaf16ac2e3015df68,
    },
    F128 {
        lo: 0x4efc8073fdf9fd3e,
        hi: 0xaf17aca93015df68,
    },
    F128 {
        lo: 0xcbfc12bbf804459a,
        hi: 0x9e24a2c3713115c2,
    },
    F128 {
        lo: 0xcafc12bae805040e,
        hi: 0x9e25a244713115c2,
    },
    F128 {
        lo: 0xe44ea287e678d918,
        hi: 0xa8a2af0500b89c58,
    },
    F128 {
        lo: 0xe54ea286f679988c,
        hi: 0xa8a3af8200b89c58,
    },
    F128 {
        lo: 0x604e304ef3842028,
        hi: 0x9990a1e8419c56f2,
    },
    F128 {
        lo: 0x614e304fe38561bc,
        hi: 0x9991a16f419c56f2,
    },
    F128 {
        lo: 0x7f1e7232ee8d3a77,
        hi: 0xe3051eb5ec3ef4d6,
    },
    F128 {
        lo: 0x7e1e7233fe8c7be3,
        hi: 0xe3041e32ec3ef4d6,
    },
    F128 {
        lo: 0xfb1ee0fbfb71c347,
        hi: 0xd2371058ad1a3e7c,
    },
    F128 {
        lo: 0xfa1ee0faeb7082d3,
        hi: 0xd23610dfad1a3e7c,
    },
    F128 {
        lo: 0x2ec7b759f38e62a9,
        hi: 0x9cffa9012454ae0c,
    },
    F128 {
        lo: 0x2fc7b758e38f233d,
        hi: 0x9cfea9862454ae0c,
    },
    F128 {
        lo: 0xaac72590e6729b99,
        hi: 0xadcda7ec657064a6,
    },
    F128 {
        lo: 0xabc72591f673da0d,
        hi: 0xadcca76b657064a6,
    },
    F128 {
        lo: 0xb59767ecfb7b81c6,
        hi: 0xd75818b1c8d2c682,
    },
    F128 {
        lo: 0xb49767edeb7ac052,
        hi: 0xd7591836c8d2c682,
    },
    F128 {
        lo: 0x3197f525ee8778f6,
        hi: 0xe66a165c89f60c28,
    },
    F128 {
        lo: 0x3097f524fe863962,
        hi: 0xe66b16db89f60c28,
    },
    F128 {
        lo: 0x1e254519f0fbe474,
        hi: 0xd0ec1b9af87f85b2,
    },
    F128 {
        lo: 0x1f254518e0faa5e0,
        hi: 0xd0ed1b1df87f85b2,
    },
    F128 {
        lo: 0x9a25d7d0e5071d44,
        hi: 0xe1de1577b95b4f18,
    },
    F128 {
        lo: 0x9b25d7d1f5065cd0,
        hi: 0xe1df15f0b95b4f18,
    },
    F128 {
        lo: 0x857595acf80e071b,
        hi: 0x9b4baa2a14f9ed3c,
    },
    F128 {
        lo: 0x847595ade80f468f,
        hi: 0x9b4aaaad14f9ed3c,
    },
    F128 {
        lo: 0x01750765edf2fe2b,
        hi: 0xaa79a4c755dd2796,
    },
    F128 {
        lo: 0x00750764fdf3bfbf,
        hi: 0xaa78a44055dd2796,
    },
    F128 {
        lo: 0x6fb081822d5a5bf0,
        hi: 0xd7d04dd1b61cbe98,
    },
    F128 {
        lo: 0x6eb081833d5b1a64,
        hi: 0xd7d14d56b61cbe98,
    },
    F128 {
        lo: 0xebb0134b38a6a2c0,
        hi: 0xe6e2433cf7387432,
    },
    F128 {
        lo: 0xeab0134a28a7e354,
        hi: 0xe6e343bbf7387432,
    },
    F128 {
        lo: 0xf4e0513725afb89f,
        hi: 0x9c77fc615a9ad616,
    },
    F128 {
        lo: 0xf5e0513635aef90b,
        hi: 0x9c76fce65a9ad616,
    },
    F128 {
        lo: 0x70e0c3fe305341af,
        hi: 0xad45f28c1bbe1cbc,
    },
    F128 {
        lo: 0x71e0c3ff2052003b,
        hi: 0xad44f20b1bbe1cbc,
    },
    F128 {
        lo: 0x5f5273c22e2fdd2d,
        hi: 0x9bc3ff4a6a379526,
    },
    F128 {
        lo: 0x5e5273c33e2e9cb9,
        hi: 0x9bc2ffcd6a379526,
    },
    F128 {
        lo: 0xdb52e10b3bd3241d,
        hi: 0xaaf1f1a72b135f8c,
    },
    F128 {
        lo: 0xda52e10a2bd26589,
        hi: 0xaaf0f1202b135f8c,
    },
    F128 {
        lo: 0xc402a37726da3e42,
        hi: 0xd0644efa86b1fda8,
    },
    F128 {
        lo: 0xc502a37636db7fd6,
        hi: 0xd0654e7d86b1fda8,
    },
    F128 {
        lo: 0x400231be3326c772,
        hi: 0xe1564017c7953702,
    },
    F128 {
        lo: 0x410231bf232786e6,
        hi: 0xe1574090c7953702,
    },
    F128 {
        lo: 0xbb1cd145c8570435,
        hi: 0x3361504f6a8f097e,
    },
    F128 {
        lo: 0xba1cd144d85645a1,
        hi: 0x336050c86a8f097e,
    },
    F128 {
        lo: 0x3f1c438cddabfd05,
        hi: 0x02535ea22babc3d4,
    },
    F128 {
        lo: 0x3e1c438dcdaabc91,
        hi: 0x02525e252babc3d4,
    },
    F128 {
        lo: 0x204c01f0c0a2e75a,
        hi: 0x78c6e1ff860961f0,
    },
    F128 {
        lo: 0x214c01f1d0a3a6ce,
        hi: 0x78c7e178860961f0,
    },
    F128 {
        lo: 0xa44c9339d55e1e6a,
        hi: 0x49f4ef12c72dab5a,
    },
    F128 {
        lo: 0xa54c9338c55f5ffe,
        hi: 0x49f5ef95c72dab5a,
    },
    F128 {
        lo: 0x8bfe2305cb2282e8,
        hi: 0x7f72e2d4b6a422c0,
    },
    F128 {
        lo: 0x8afe2304db23c37c,
        hi: 0x7f73e253b6a422c0,
    },
    F128 {
        lo: 0x0ffeb1ccdede7bd8,
        hi: 0x4e40ec39f780e86a,
    },
    F128 {
        lo: 0x0efeb1cdcedf3a4c,
        hi: 0x4e41ecbef780e86a,
    },
    F128 {
        lo: 0x10aef3b0c3d76187,
        hi: 0x34d553645a224a4e,
    },
    F128 {
        lo: 0x11aef3b1d3d62013,
        hi: 0x34d453e35a224a4e,
    },
    F128 {
        lo: 0x94ae6179d62b98b7,
        hi: 0x05e75d891b0680e4,
    },
    F128 {
        lo: 0x95ae6178c62ad923,
        hi: 0x05e65d0e1b0680e4,
    },
    F128 {
        lo: 0xfa6be79e16833d6c,
        hi: 0x784eb49ff8c719ea,
    },
    F128 {
        lo: 0xfb6be79f06827cf8,
        hi: 0x784fb418f8c719ea,
    },
    F128 {
        lo: 0x7e6b7557037fc45c,
        hi: 0x497cba72b9e3d340,
    },
    F128 {
        lo: 0x7f6b7556137e85c8,
        hi: 0x497dbaf5b9e3d340,
    },
    F128 {
        lo: 0x613b372b1e76de03,
        hi: 0x33e9052f14417164,
    },
    F128 {
        lo: 0x603b372a0e779f97,
        hi: 0x33e805a814417164,
    },
    F128 {
        lo: 0xe53ba5e20b8a2733,
        hi: 0x02db0bc25565bbce,
    },
    F128 {
        lo: 0xe43ba5e31b8b66a7,
        hi: 0x02da0b455565bbce,
    },
    F128 {
        lo: 0xca8915de15f6bbb1,
        hi: 0x345d060424ec3254,
    },
    F128 {
        lo: 0xcb8915df05f7fa25,
        hi: 0x345c068324ec3254,
    },
    F128 {
        lo: 0x4e898717000a4281,
        hi: 0x056f08e965c8f8fe,
    },
    F128 {
        lo: 0x4f898716100b0315,
        hi: 0x056e086e65c8f8fe,
    },
    F128 {
        lo: 0x51d9c56b1d0358de,
        hi: 0x7ffab7b4c86a5ada,
    },
    F128 {
        lo: 0x50d9c56a0d02194a,
        hi: 0x7ffbb733c86a5ada,
    },
    F128 {
        lo: 0xd5d957a208ffa1ee,
        hi: 0x4ec8b959894e9070,
    },
    F128 {
        lo: 0xd4d957a318fee07a,
        hi: 0x4ec9b9de894e9070,
    },
];

const fn mul_by_x(value: F128) -> F128 {
    let carry = value.hi >> 63;
    let mask = 0u64.wrapping_sub(carry);
    F128 {
        lo: (value.lo << 1) ^ (0x87 & mask),
        hi: (value.hi << 1) | (value.lo >> 63),
    }
}

const fn build_tab8() -> [F128; TAB8_LEN] {
    let mut out = [F128::ZERO; TAB8_LEN];
    let mut t = 0;
    while t < TWIDDLE_COUNT {
        let mut powers = [F128::ZERO; 8];
        powers[0] = TWIDDLES[t];
        let mut bit = 1;
        while bit < 8 {
            powers[bit] = mul_by_x(powers[bit - 1]);
            bit += 1;
        }
        let mut byte = 0;
        while byte < 256 {
            let mut value = F128::ZERO;
            bit = 0;
            while bit < 8 {
                if byte & (1 << bit) != 0 {
                    value.lo ^= powers[bit].lo;
                    value.hi ^= powers[bit].hi;
                }
                bit += 1;
            }
            out[t * TAB8_PER_TWIDDLE + byte] = value;
            byte += 1;
        }
        t += 1;
    }
    out
}

pub(super) static TAB8: [F128; TAB8_LEN] = build_tab8();
