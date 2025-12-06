# flow-matching
```mermaid
graph TD
    subgraph Input Streams
        A[Text Prompt] --> B(Text Encoder);
        C[Noisy Image Latents] --> D(Image Embedding);
    end

    B --> E(Text Tokens);
    D --> F(Image Tokens);

    subgraph MM-DiT Block (Repeated N times)
        G((Join Tokens))
        E --> G;
        F --> G;
        
        G --> H{Joint Attention Layer};
        H --> I(Refined Tokens);
        I --> J[Feed Forward (MLP)];
    end

    J --> K[Final Latent Output];
```
