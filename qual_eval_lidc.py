#######################
# Qualitative Testing #
#######################

if True:

    rng = np.random.default_rng(seed=42)

    NUM_IMGS = 5
    NUM_SAMPLES = 5
    PROB_ISOVAL = 0.8

    # Load model
    checkpoint_path = Path("training_logs/NicolasWork-lidc-patches-lv3/model_checkpoints//checkpoint_epoch-40.pth")
    state_dict = torch.load(str(checkpoint_path), map_location=torch.device(DEVICE))

    net = ProbabilisticUnet(num_input_channels=1,
                            num_classes=NUM_CLASSES,
                            num_filters=NUM_CHANNELS,
                            latent_dim=LATENT_DIM,
                            no_convs_fcomb=NUM_CONVS_FCOMB,
                            beta=BETA,
                            device=DEVICE)
    net.to(DEVICE)

    net.load_state_dict(state_dict=state_dict)

    # Get random images
    test_dataset = LIDCCrops(data_home="data/lidc_crops", split="test", transform=dict(resize=dict(output_size=(128, 128))))
    metadatas, imgs, segs = zip(*[test_dataset[i] for i in rng.choice(np.arange(len(test_dataset)), NUM_IMGS, replace=False)])
    
    imgs = [img.unsqueeze(dim=0) for img in imgs]    
    imgs = torch.cat(imgs, dim=0)

    segs = [seg.unsqueeze(dim=0).unsqueeze(dim=0) for seg in segs]    
    segs = torch.cat(segs, dim=0)

    imgs = imgs.to(DEVICE)

    probs = []
    preds = []
    for nsample in range(NUM_SAMPLES):
        net(imgs, None, training=False)  # Run net (this initializes the unet features and the latent space)
        prob = net.sample(testing=True)  # samples a segmentation using the unet features + the latent space
        
        pred = prob > PROB_ISOVAL

        # Use when num classes > 1
        # prob = nn.Softmax(dim=1)(sample)
        # pred = torch.argmax(probs, dim=1)

        probs.append(prob)
        preds.append(pred)


    # plot
    map_to_vis = [probs, preds][0]

    fig, axs = plt.subplots(nrows=NUM_IMGS, ncols=NUM_SAMPLES+2, layout="tight")

    # - plot imgs
    for img_i in range(NUM_IMGS):
        axs[img_i, 0].imshow(imgs.cpu().numpy()[img_i, 0], cmap="gray")
        axs[img_i, 0].set_ylabel(f"Image {img_i}")

    # - plot gt
    axs[0, 1].set_title("GT 1/4")
    for img_i in range(NUM_IMGS):
        axs[img_i, 1].imshow(segs.cpu().numpy()[img_i, 0], cmap="gray")        

    # - plot samples
    for sample_i in range(NUM_SAMPLES):
        axs[0, sample_i + 2].set_title(f"Sample {sample_i}")
        for img_i in range(NUM_IMGS):
            axs[img_i, sample_i + 2].imshow(map_to_vis[sample_i].detach().cpu().numpy()[img_i, 0])

    for ax in axs.flatten():
        # Hide X and Y axes label marks
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)

        # Hide X and Y axes tick marks
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()