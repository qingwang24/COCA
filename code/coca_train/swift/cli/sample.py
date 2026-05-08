# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
custom_swift_path = "/dmx-csy-mix01/cog3/permanent/ycpan4/Projects/GeoRL/v1"
if custom_swift_path not in sys.path:
    sys.path.insert(0, custom_swift_path)
if __name__ == '__main__':
    from swift.ray import try_init_ray
    try_init_ray()
    from swift.llm.sampling import sampling_main
    sampling_main()
