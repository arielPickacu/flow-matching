# flow-matching
```
graph TD
    subgraph "3D Input"
        Video[Video Latents (H x W x Time)] --> Patches[3D Patches]
        Patches --> Linear[Linear Projection + 3D RoPE]
    end

    Linear --> Tokens[Video Tokens]

    subgraph "ST-DiT Block (Repeated N times)"
        direction TB
        Tokens --> S_Attn[Spatial Attention]
        note1[("Look at pixels in\nCURRENT frame only")] 
        S_Attn -.- note1
        
        S_Attn --> T_Attn[Temporal Attention]
        note2[("Look at same pixel\nacross ALL frames")]
        T_Attn -.- note2
        
        T_Attn --> MLP[Feed Forward Network]
    end

    MLP --> Unpatch[Unpatchify]
    Unpatch --> Output[Generated Video]
```
