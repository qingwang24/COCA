# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
custom_swift_path = "/dmx-csy-mix01/cog3/permanent/qkchang/shliu19/critique/code/GRPO/swift_0707_qwen2.5vl_actor/"
if custom_swift_path not in sys.path:
    sys.path.insert(0, custom_swift_path)
    
if __name__ == '__main__':
    from swift.cli.utils import try_use_single_device_mode
    try_use_single_device_mode()
    from swift.llm import rlhf_main
    rlhf_main()
