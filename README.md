# NeuralZ

This project is a modernization of the RocAlphaGo project that seems dead. It was originally intended to re-implement AlphaGo. Mostly only the policy network was fully implemented. The value and rollout networks had some work done but computational challenges prevented them from being fully utilized.

In 2016 a policy network was trained using a conventional cnn of 12 layers and 96 filters using GoGoD data. This used less filters than the original AlphaGo due to hardware constraints. It was run on KGS as a bot and seemed to achieve about 2k strength.

In 2017 a policy network was trained using a conventional cnn of 12 layers and 96 filters using KGS data. This mostly recreated the AlphaGo policy network. It was run on KGS as a bot in 2022 and seemed to achieve about 2d strength.

In 2026 this project was created. The original project was updated to use uv and docker, upgraded all of the libraries and began reproduction of the 2017 run against modern hardware. The 2017 run was trained on 2 months with a gtx 1060. A resnet tower model (b10c128) was added to this project and trained on katago selfplay data. After 22 hours of training on an rtx 4070 super it was run on KGS as a bot and appears to achieve around 4d strength.

Currently training a b15c192 model which is converging much faster despite the longer training time per step.