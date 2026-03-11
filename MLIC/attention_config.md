# MLIC++ Attention Module Configuration

The `MLICPlusPlus` model supports enabling or disabling its attention modules via configuration. The following options can be set in your config (YAML, dict, etc.):

- `enable_channel_context`: Enable/disable channel context attention (default: True)
- `enable_local_context`: Enable/disable local context attention (default: True)
- `enable_global_inter_context`: Enable/disable global inter-slice attention (default: True)
- `enable_global_intra_context`: Enable/disable global intra-slice attention (default: True)

## Example

```yaml
model:
  name: MLICPlusPlus
  N: 192
  M: 320
  slice_num: 10
  context_window: 5
  enable_channel_context: true
  enable_local_context: false
  enable_global_inter_context: true
  enable_global_intra_context: false
```

If a flag is set to `false`, the corresponding attention module will be bypassed in the model's forward, compress, and decompress passes.
