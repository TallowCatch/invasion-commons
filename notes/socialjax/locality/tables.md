
### Locality per testbed and layout (scripted bots, pooled over the 4 populations)

| Testbed | Regions | χ (range over populations) | violations | ζ footprint | ζ static | ζ counterfactual | S_10 (formula (1−χ)^9) | ξ_loc w = 1 | w = 10 | w = 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| Harvest (7) | strips R=2 | 0.0279 (0.0179–0.0391) | 827 | 1.000 | 1.000 | 1.000 | 0.893 (0.775) | 0.000 | 0.107 | 0.698 |
| Harvest (7) | strips R=4 | 0.0725 (0.0625–0.0839) | 945 | 0.967 | 0.965 | 0.973 | 0.585 (0.508) | 0.033 | 0.426 | 0.845 |
| Harvest (7) | bands R=2 | 0.0283 (0.0217–0.0341) | 939 | 0.909 | 0.913 | 0.860 | 0.792 (0.773) | 0.091 | 0.243 | 0.609 |
| Harvest (7) | grid R=4 | 0.0493 (0.0452–0.0543) | 912 | 0.896 | 0.903 | 0.831 | 0.654 (0.634) | 0.104 | 0.375 | 0.898 |
| Harvest, large (14) | strips R=2 | 0.0171 (0.0085–0.0248) | 689 | 0.971 | 0.971 | 0.978 | 0.894 (0.857) | 0.029 | 0.120 | 0.528 |
| Harvest, large (14) | strips R=4 | 0.0397 (0.0296–0.0455) | 869 | 0.974 | 0.974 | 0.976 | 0.779 (0.694) | 0.026 | 0.234 | 0.842 |
| Harvest, large (14) | grid R=4 | 0.0424 (0.0393–0.0480) | 899 | 0.863 | 0.871 | 0.722 | 0.669 (0.677) | 0.137 | 0.369 | 0.806 |
| Rooms, closed (7) | rooms (R=2) | 0.0000 (0.0000–0.0000) | 391 | 1.000 | 1.000 | 1.000 | 1.000 (1.000) | 0.000 | 0.000 | 0.000 |
| Rooms, closed (7) | strips R=4 | 0.0421 (0.0403–0.0440) | 725 | 0.971 | 0.975 | 0.977 | 0.811 (0.679) | 0.029 | 0.203 | 0.746 |
| Rooms, one door (7) | rooms (R=2) | 0.0062 (0.0019–0.0136) | 749 | 1.000 | 1.000 | 1.000 | 1.000 (0.946) | 0.000 | 0.000 | 0.402 |
| Rooms, one door (7) | strips R=4 | 0.0416 (0.0334–0.0455) | 919 | 0.971 | 0.972 | 0.980 | 0.795 (0.682) | 0.029 | 0.218 | 0.810 |
| Rooms, large (14) | rooms (R=4) | 0.0000 (0.0000–0.0000) | 331 | 0.997 | 0.996 | 0.994 | 1.000 (1.000) | 0.003 | 0.003 | 0.003 |
| Rooms, large (14) | strips R=2 | 0.0000 (0.0000–0.0000) | 371 | 0.985 | 0.990 | 0.992 | 1.000 (1.000) | 0.015 | 0.015 | 0.015 |
| Coins (2) | bands R=2 | 0.0496 (0.0446–0.0621) | 418 | 0.665 | – | – | 0.634 (0.632) | 0.335 | 0.562 | 0.975 |
| Coins (2) | strips R=2 | 0.0529 (0.0508–0.0580) | 419 | 0.652 | – | – | 0.623 (0.613) | 0.348 | 0.597 | 0.988 |
| Clean Up (7) | strips R=2 | 0.0151 (0.0090–0.0225) | 733 | 0.504 | – | – | 0.824 (0.872) | 0.496 | 0.586 | 0.848 |
| Clean Up (7) | strips R=4 | 0.0375 (0.0279–0.0432) | 742 | 0.257 | – | – | 0.538 (0.709) | 0.743 | 0.866 | 0.994 |
| Clean Up (7) | bands R=2 | 0.0276 (0.0184–0.0486) | 733 | 1.000 | – | – | 1.000 (0.777) | 0.000 | 0.000 | 0.000 |

### Per population (χ; S_10 measured vs (1 − χ)^9)

| Testbed | Regions | Population | χ | violations | ζ | S_10 | (1−χ)^9 | S_50 | (1−χ)^49 |
|---|---|---|---|---|---|---|---|---|---|
| Harvest (7) | strips R=2 | 7 honest | 0.0265 | 29 | 1.000 | 0.929 | 0.785 | 0.417 | 0.268 |
| Harvest (7) | strips R=2 | 1 greedy + 6 honest | 0.0179 | 277 | 1.000 | 0.968 | 0.850 | 0.545 | 0.414 |
| Harvest (7) | strips R=2 | 7 sneaky | 0.0280 | 191 | 1.000 | 0.737 | 0.775 | 0.247 | 0.249 |
| Harvest (7) | strips R=2 | 7 migrator | 0.0391 | 330 | 1.000 | 0.918 | 0.698 | 0.119 | 0.142 |
| Harvest (7) | strips R=4 | 7 honest | 0.0839 | 29 | 1.000 | 0.429 | 0.454 | 0.042 | 0.014 |
| Harvest (7) | strips R=4 | 1 greedy + 6 honest | 0.0625 | 277 | 0.965 | 0.643 | 0.560 | 0.242 | 0.042 |
| Harvest (7) | strips R=4 | 7 sneaky | 0.0806 | 350 | 0.969 | 0.578 | 0.469 | 0.076 | 0.016 |
| Harvest (7) | strips R=4 | 7 migrator | 0.0631 | 289 | 0.963 | 0.554 | 0.556 | 0.183 | 0.041 |
| Harvest (7) | bands R=2 | 7 honest | 0.0217 | 29 | 0.949 | 0.857 | 0.821 | 0.583 | 0.341 |
| Harvest (7) | bands R=2 | 1 greedy + 6 honest | 0.0273 | 277 | 0.899 | 0.675 | 0.779 | 0.116 | 0.257 |
| Harvest (7) | bands R=2 | 7 sneaky | 0.0341 | 328 | 0.929 | 0.848 | 0.732 | 0.539 | 0.183 |
| Harvest (7) | bands R=2 | 7 migrator | 0.0300 | 305 | 0.892 | 0.833 | 0.760 | 0.505 | 0.225 |
| Harvest (7) | grid R=4 | 7 honest | 0.0482 | 29 | 0.949 | 0.786 | 0.641 | 0.167 | 0.089 |
| Harvest (7) | grid R=4 | 1 greedy + 6 honest | 0.0452 | 277 | 0.899 | 0.643 | 0.659 | 0.079 | 0.104 |
| Harvest (7) | grid R=4 | 7 sneaky | 0.0543 | 309 | 0.882 | 0.570 | 0.605 | 0.129 | 0.065 |
| Harvest (7) | grid R=4 | 7 migrator | 0.0496 | 297 | 0.903 | 0.741 | 0.633 | 0.101 | 0.083 |
| Harvest, large (14) | strips R=2 | 14 honest | 0.0184 | 21 | 0.968 | 0.952 | 0.846 | 0.619 | 0.402 |
| Harvest, large (14) | strips R=2 | 1 greedy + 13 honest | 0.0085 | 269 | 0.976 | 0.874 | 0.926 | 0.550 | 0.658 |
| Harvest, large (14) | strips R=2 | 14 sneaky | 0.0165 | 116 | 0.956 | 0.879 | 0.861 | 0.605 | 0.442 |
| Harvest, large (14) | strips R=2 | 14 migrator | 0.0248 | 283 | 0.973 | 0.915 | 0.798 | 0.346 | 0.292 |
| Harvest, large (14) | strips R=4 | 14 honest | 0.0395 | 21 | 0.968 | 0.667 | 0.696 | 0.190 | 0.139 |
| Harvest, large (14) | strips R=4 | 1 greedy + 13 honest | 0.0296 | 269 | 0.976 | 0.836 | 0.763 | 0.219 | 0.230 |
| Harvest, large (14) | strips R=4 | 14 sneaky | 0.0455 | 284 | 0.965 | 0.661 | 0.658 | 0.167 | 0.102 |
| Harvest, large (14) | strips R=4 | 14 migrator | 0.0443 | 295 | 0.982 | 0.847 | 0.665 | 0.092 | 0.109 |
| Harvest, large (14) | grid R=4 | 14 honest | 0.0407 | 21 | 0.894 | 0.810 | 0.688 | 0.286 | 0.131 |
| Harvest, large (14) | grid R=4 | 1 greedy + 13 honest | 0.0393 | 269 | 0.867 | 0.617 | 0.697 | 0.104 | 0.140 |
| Harvest, large (14) | grid R=4 | 14 sneaky | 0.0480 | 325 | 0.857 | 0.655 | 0.642 | 0.277 | 0.090 |
| Harvest, large (14) | grid R=4 | 14 migrator | 0.0416 | 284 | 0.865 | 0.722 | 0.682 | 0.211 | 0.125 |
| Rooms, closed (7) | rooms (R=2) | 7 honest | 0.0000 | 28 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Rooms, closed (7) | rooms (R=2) | 1 greedy + 6 honest | 0.0000 | 159 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Rooms, closed (7) | rooms (R=2) | 7 sneaky | 0.0000 | 102 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Rooms, closed (7) | rooms (R=2) | 7 migrator | 0.0000 | 102 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Rooms, closed (7) | strips R=4 | 7 honest | 0.0403 | 28 | 0.993 | 0.821 | 0.691 | 0.269 | 0.134 |
| Rooms, closed (7) | strips R=4 | 1 greedy + 6 honest | 0.0411 | 159 | 0.972 | 0.767 | 0.685 | 0.278 | 0.128 |
| Rooms, closed (7) | strips R=4 | 7 sneaky | 0.0429 | 221 | 0.971 | 0.819 | 0.674 | 0.258 | 0.116 |
| Rooms, closed (7) | strips R=4 | 7 migrator | 0.0440 | 317 | 0.967 | 0.826 | 0.667 | 0.241 | 0.110 |
| Rooms, one door (7) | rooms (R=2) | 7 honest | 0.0030 | 27 | 1.000 | 1.000 | 0.973 | 0.889 | 0.864 |
| Rooms, one door (7) | rooms (R=2) | 1 greedy + 6 honest | 0.0019 | 287 | 1.000 | 1.000 | 0.983 | 0.676 | 0.909 |
| Rooms, one door (7) | rooms (R=2) | 7 sneaky | 0.0062 | 206 | 1.000 | 1.000 | 0.946 | 0.641 | 0.738 |
| Rooms, one door (7) | rooms (R=2) | 7 migrator | 0.0136 | 229 | 1.000 | 1.000 | 0.884 | 0.427 | 0.511 |
| Rooms, one door (7) | strips R=4 | 7 honest | 0.0334 | 27 | 0.993 | 0.667 | 0.736 | 0.259 | 0.189 |
| Rooms, one door (7) | strips R=4 | 1 greedy + 6 honest | 0.0454 | 287 | 0.964 | 0.774 | 0.659 | 0.233 | 0.103 |
| Rooms, one door (7) | strips R=4 | 7 sneaky | 0.0419 | 311 | 0.979 | 0.781 | 0.680 | 0.227 | 0.123 |
| Rooms, one door (7) | strips R=4 | 7 migrator | 0.0455 | 294 | 0.968 | 0.844 | 0.657 | 0.116 | 0.102 |
| Rooms, large (14) | rooms (R=4) | 14 honest | 0.0000 | 32 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Rooms, large (14) | rooms (R=4) | 1 greedy + 13 honest | 0.0000 | 101 | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 |
| Rooms, large (14) | rooms (R=4) | 14 sneaky | 0.0000 | 99 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Rooms, large (14) | rooms (R=4) | 14 migrator | 0.0000 | 99 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Rooms, large (14) | strips R=2 | 14 honest | 0.0000 | 32 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Rooms, large (14) | strips R=2 | 1 greedy + 13 honest | 0.0000 | 101 | 0.990 | 1.000 | 1.000 | 1.000 | 1.000 |
| Rooms, large (14) | strips R=2 | 14 sneaky | 0.0000 | 119 | 0.981 | 1.000 | 1.000 | 1.000 | 1.000 |
| Rooms, large (14) | strips R=2 | 14 migrator | 0.0000 | 119 | 0.981 | 1.000 | 1.000 | 1.000 | 1.000 |
| Coins (2) | bands R=2 | 2 honest | 0.0446 | 7 | 0.429 | 0.714 | 0.663 | 0.167 | 0.107 |
| Coins (2) | bands R=2 | 1 greedy + 1 honest | 0.0471 | 356 | 0.680 | 0.645 | 0.647 | 0.029 | 0.094 |
| Coins (2) | bands R=2 | 2 sneaky | 0.0446 | 7 | 0.429 | 0.714 | 0.663 | 0.167 | 0.107 |
| Coins (2) | bands R=2 | 2 migrator | 0.0621 | 48 | 0.625 | 0.532 | 0.562 | 0.023 | 0.043 |
| Coins (2) | strips R=2 | 2 honest | 0.0508 | 7 | 0.714 | 0.714 | 0.625 | 0.000 | 0.078 |
| Coins (2) | strips R=2 | 1 greedy + 1 honest | 0.0520 | 356 | 0.666 | 0.636 | 0.619 | 0.035 | 0.073 |
| Coins (2) | strips R=2 | 2 sneaky | 0.0508 | 10 | 0.500 | 0.700 | 0.625 | 0.000 | 0.078 |
| Coins (2) | strips R=2 | 2 migrator | 0.0580 | 46 | 0.565 | 0.489 | 0.584 | 0.000 | 0.054 |
| Clean Up (7) | strips R=2 | 7 honest | 0.0145 | 0 | – | – | 0.877 | – | 0.489 |
| Clean Up (7) | strips R=2 | 1 greedy + 6 honest | 0.0090 | 733 | 0.504 | 0.824 | 0.922 | 0.303 | 0.644 |
| Clean Up (7) | strips R=2 | 7 sneaky | 0.0143 | 0 | – | – | 0.878 | – | 0.493 |
| Clean Up (7) | strips R=2 | 7 migrator | 0.0225 | 0 | – | – | 0.815 | – | 0.328 |
| Clean Up (7) | strips R=4 | 7 honest | 0.0408 | 0 | – | – | 0.687 | – | 0.130 |
| Clean Up (7) | strips R=4 | 1 greedy + 6 honest | 0.0279 | 733 | 0.256 | 0.540 | 0.775 | 0.027 | 0.250 |
| Clean Up (7) | strips R=4 | 7 sneaky | 0.0381 | 2 | 0.264 | 0.500 | 0.705 | 0.000 | 0.149 |
| Clean Up (7) | strips R=4 | 7 migrator | 0.0432 | 7 | 0.296 | 0.286 | 0.672 | 0.143 | 0.115 |
| Clean Up (7) | bands R=2 | 7 honest | 0.0217 | 0 | – | – | 0.821 | – | 0.341 |
| Clean Up (7) | bands R=2 | 1 greedy + 6 honest | 0.0184 | 733 | 1.000 | 1.000 | 0.846 | 1.000 | 0.403 |
| Clean Up (7) | bands R=2 | 7 sneaky | 0.0217 | 0 | – | – | 0.821 | – | 0.341 |
| Clean Up (7) | bands R=2 | 7 migrator | 0.0486 | 0 | – | – | 0.638 | – | 0.087 |

### Task 2, abstract model: largest |simulation − formula| over the grid

| Crossing pattern | cells | simple P_0 formula: worst error | Markov P_0: worst error |
|---|---|---|---|
| drift | 180 | 0.002 (q = 0.286, χ = 0.01, w = 5, α = 0.0) | 0.0024 |
| two | 180 | 0.727 (q = 0.143, χ = 0.5, w = 50, α = 0.0) | 0.0025 |
| ring4 | 180 | 0.598 (q = 0.143, χ = 0.5, w = 50, α = 0.0) | 0.0022 |

### Task 2, real overseer code (corridor, N = 8, K = 2): catch share

| R | w | χ | simulated | P_C | simple P_0 | Markov P_0 (q = K/N) | Markov P_0 (measured q) |
|---|---|---|---|---|---|---|---|
| 2 | 1 | 0.0 | 0.254 | 0.250 | 0.250 | 0.250 | 0.250 |
| 2 | 1 | 0.01 | 0.250 | 0.250 | 0.250 | 0.250 | 0.249 |
| 2 | 1 | 0.05 | 0.250 | 0.250 | 0.250 | 0.250 | 0.249 |
| 2 | 1 | 0.2 | 0.253 | 0.250 | 0.250 | 0.250 | 0.249 |
| 2 | 1 | 0.5 | 0.251 | 0.250 | 0.250 | 0.250 | 0.249 |
| 2 | 5 | 0.0 | 0.763 | 0.763 | 0.763 | 0.763 | 0.763 |
| 2 | 5 | 0.01 | 0.721 | 0.763 | 0.752 | 0.752 | 0.750 |
| 2 | 5 | 0.05 | 0.690 | 0.763 | 0.710 | 0.713 | 0.711 |
| 2 | 5 | 0.2 | 0.615 | 0.763 | 0.576 | 0.619 | 0.618 |
| 2 | 5 | 0.5 | 0.565 | 0.763 | 0.397 | 0.560 | 0.559 |
| 2 | 10 | 0.0 | 0.945 | 0.944 | 0.944 | 0.944 | 0.944 |
| 2 | 10 | 0.01 | 0.897 | 0.944 | 0.921 | 0.922 | 0.922 |
| 2 | 10 | 0.05 | 0.834 | 0.944 | 0.840 | 0.859 | 0.858 |
| 2 | 10 | 0.2 | 0.779 | 0.944 | 0.621 | 0.782 | 0.781 |
| 2 | 10 | 0.5 | 0.778 | 0.944 | 0.400 | 0.775 | 0.773 |
| 2 | 50 | 0.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 50 | 0.01 | 0.981 | 1.000 | 0.971 | 0.981 | 0.981 |
| 2 | 50 | 0.05 | 0.974 | 1.000 | 0.870 | 0.979 | 0.979 |
| 2 | 50 | 0.2 | 0.996 | 1.000 | 0.625 | 0.997 | 0.997 |
| 2 | 50 | 0.5 | 0.999 | 1.000 | 0.400 | 0.999 | 0.999 |
| 4 | 1 | 0.0 | 0.261 | 0.250 | 0.250 | 0.250 | 0.250 |
| 4 | 1 | 0.01 | 0.219 | 0.250 | 0.250 | 0.250 | 0.228 |
| 4 | 1 | 0.05 | 0.232 | 0.250 | 0.250 | 0.250 | 0.226 |
| 4 | 1 | 0.2 | 0.233 | 0.250 | 0.250 | 0.250 | 0.225 |
| 4 | 1 | 0.5 | 0.219 | 0.250 | 0.250 | 0.250 | 0.225 |
| 4 | 5 | 0.0 | 0.770 | 0.763 | 0.763 | 0.763 | 0.763 |
| 4 | 5 | 0.01 | 0.654 | 0.763 | 0.752 | 0.752 | 0.715 |
| 4 | 5 | 0.05 | 0.637 | 0.763 | 0.710 | 0.712 | 0.673 |
| 4 | 5 | 0.2 | 0.548 | 0.763 | 0.576 | 0.598 | 0.559 |
| 4 | 5 | 0.5 | 0.435 | 0.763 | 0.397 | 0.479 | 0.443 |
| 4 | 10 | 0.0 | 0.943 | 0.944 | 0.944 | 0.944 | 0.944 |
| 4 | 10 | 0.01 | 0.842 | 0.944 | 0.921 | 0.922 | 0.902 |
| 4 | 10 | 0.05 | 0.785 | 0.944 | 0.840 | 0.850 | 0.824 |
| 4 | 10 | 0.2 | 0.659 | 0.944 | 0.621 | 0.704 | 0.669 |
| 4 | 10 | 0.5 | 0.569 | 0.944 | 0.400 | 0.610 | 0.572 |
| 4 | 50 | 0.0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | 50 | 0.01 | 0.966 | 1.000 | 0.971 | 0.976 | 0.973 |
| 4 | 50 | 0.05 | 0.913 | 1.000 | 0.870 | 0.934 | 0.925 |
| 4 | 50 | 0.2 | 0.914 | 1.000 | 0.625 | 0.935 | 0.921 |
| 4 | 50 | 0.5 | 0.946 | 1.000 | 0.400 | 0.959 | 0.945 |

R = 2: worst |sim − Markov(measured q)| = 0.029; worst |sim − simple P_0| = 0.599

R = 4: worst |sim − Markov(measured q)| = 0.061; worst |sim − simple P_0| = 0.546

### Task 3: decentralisation gap Δ_dec (share of violations caught: central − regional)

| Testbed | Regions | w | Population | records | central | regional | Δ_dec (95% CI) | Δη | ξ_loc |
|---|---|---|---|---|---|---|---|---|---|
| Harvest (7) | grid R=4 | 1 | 1 greedy + 6 honest | handover | 0.255 | 0.161 | 0.094 (0.061, 0.126) | 0.093 | 0.101 |
| Harvest (7) | grid R=4 | 1 | 1 greedy + 6 honest | local | 0.255 | 0.161 | 0.094 (0.061, 0.126) | 0.093 | 0.101 |
| Harvest (7) | grid R=4 | 10 | 1 greedy + 6 honest | handover | 0.983 | 0.798 | 0.185 (0.128, 0.242) | 0.184 | 0.374 |
| Harvest (7) | grid R=4 | 10 | 1 greedy + 6 honest | local | 0.983 | 0.702 | 0.281 (0.195, 0.369) | 0.197 | 0.374 |
| Harvest (7) | grid R=4 | 50 | 1 greedy + 6 honest | handover | 1.000 | 0.993 | 0.007 (0.000, 0.019) | 0.196 | 0.922 |
| Harvest (7) | grid R=4 | 50 | 1 greedy + 6 honest | local | 1.000 | 0.849 | 0.151 (0.097, 0.220) | 0.213 | 0.922 |
| Harvest (7) | grid R=4 | 1 | 7 migrator | handover | 0.159 | 0.177 | -0.018 (-0.124, 0.089) | 0.926 | 0.097 |
| Harvest (7) | grid R=4 | 1 | 7 migrator | local | 0.159 | 0.177 | -0.018 (-0.124, 0.089) | 0.926 | 0.097 |
| Harvest (7) | grid R=4 | 10 | 7 migrator | handover | 0.984 | 0.818 | 0.166 (0.109, 0.226) | 0.909 | 0.299 |
| Harvest (7) | grid R=4 | 10 | 7 migrator | local | 0.984 | 0.769 | 0.215 (0.151, 0.277) | 0.909 | 0.299 |
| Harvest (7) | grid R=4 | 50 | 7 migrator | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.908 | 0.902 |
| Harvest (7) | grid R=4 | 50 | 7 migrator | local | 1.000 | 0.942 | 0.058 (0.024, 0.095) | 0.908 | 0.902 |
| Harvest (7) | grid R=4 | 1 | 7 sneaky | handover | 0.159 | 0.127 | 0.032 (-0.056, 0.130) | 0.768 | 0.118 |
| Harvest (7) | grid R=4 | 1 | 7 sneaky | local | 0.159 | 0.127 | 0.032 (-0.056, 0.130) | 0.768 | 0.118 |
| Harvest (7) | grid R=4 | 10 | 7 sneaky | handover | 0.984 | 0.805 | 0.179 (0.140, 0.213) | 0.734 | 0.461 |
| Harvest (7) | grid R=4 | 10 | 7 sneaky | local | 0.984 | 0.653 | 0.330 (0.269, 0.387) | 0.718 | 0.461 |
| Harvest (7) | grid R=4 | 50 | 7 sneaky | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.756 | 0.878 |
| Harvest (7) | grid R=4 | 50 | 7 sneaky | local | 1.000 | 0.840 | 0.160 (0.116, 0.196) | 0.763 | 0.878 |
| Harvest (7) | strips R=2 | 1 | 1 greedy + 6 honest | handover | 0.255 | 0.302 | -0.047 (-0.111, 0.003) | -0.139 | 0.000 |
| Harvest (7) | strips R=2 | 1 | 1 greedy + 6 honest | local | 0.255 | 0.302 | -0.047 (-0.111, 0.003) | -0.139 | 0.000 |
| Harvest (7) | strips R=2 | 10 | 1 greedy + 6 honest | handover | 0.983 | 0.902 | 0.081 (0.033, 0.132) | 0.053 | 0.032 |
| Harvest (7) | strips R=2 | 10 | 1 greedy + 6 honest | local | 0.983 | 0.901 | 0.082 (0.031, 0.134) | 0.023 | 0.032 |
| Harvest (7) | strips R=2 | 50 | 1 greedy + 6 honest | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.052 | 0.455 |
| Harvest (7) | strips R=2 | 50 | 1 greedy + 6 honest | local | 1.000 | 0.976 | 0.024 (0.007, 0.048) | 0.022 | 0.455 |
| Harvest (7) | strips R=2 | 1 | 7 migrator | handover | 0.159 | 0.167 | -0.007 (-0.099, 0.081) | 0.685 | 0.000 |
| Harvest (7) | strips R=2 | 1 | 7 migrator | local | 0.159 | 0.167 | -0.007 (-0.099, 0.081) | 0.685 | 0.000 |
| Harvest (7) | strips R=2 | 10 | 7 migrator | handover | 0.984 | 0.886 | 0.097 (0.061, 0.143) | 0.660 | 0.082 |
| Harvest (7) | strips R=2 | 10 | 7 migrator | local | 0.984 | 0.886 | 0.097 (0.061, 0.143) | 0.660 | 0.082 |
| Harvest (7) | strips R=2 | 50 | 7 migrator | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.669 | 0.881 |
| Harvest (7) | strips R=2 | 50 | 7 migrator | local | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.669 | 0.881 |
| Harvest (7) | strips R=2 | 1 | 7 sneaky | handover | 0.159 | 0.209 | -0.049 (-0.150, 0.046) | 0.211 | 0.000 |
| Harvest (7) | strips R=2 | 1 | 7 sneaky | local | 0.159 | 0.209 | -0.049 (-0.150, 0.046) | 0.211 | 0.000 |
| Harvest (7) | strips R=2 | 10 | 7 sneaky | handover | 0.984 | 0.852 | 0.131 (0.048, 0.212) | 0.166 | 0.263 |
| Harvest (7) | strips R=2 | 10 | 7 sneaky | local | 0.984 | 0.852 | 0.131 (0.048, 0.212) | 0.166 | 0.263 |
| Harvest (7) | strips R=2 | 50 | 7 sneaky | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.166 | 0.753 |
| Harvest (7) | strips R=2 | 50 | 7 sneaky | local | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.166 | 0.753 |
| Harvest (7) | strips R=4 | 1 | 1 greedy + 6 honest | handover | 0.255 | 0.208 | 0.047 (0.014, 0.088) | -0.004 | 0.035 |
| Harvest (7) | strips R=4 | 1 | 1 greedy + 6 honest | local | 0.255 | 0.208 | 0.047 (0.014, 0.088) | -0.004 | 0.035 |
| Harvest (7) | strips R=4 | 10 | 1 greedy + 6 honest | handover | 0.983 | 0.836 | 0.147 (0.091, 0.198) | 0.019 | 0.366 |
| Harvest (7) | strips R=4 | 10 | 1 greedy + 6 honest | local | 0.983 | 0.815 | 0.168 (0.107, 0.229) | 0.045 | 0.366 |
| Harvest (7) | strips R=4 | 50 | 1 greedy + 6 honest | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.024 | 0.762 |
| Harvest (7) | strips R=4 | 50 | 1 greedy + 6 honest | local | 1.000 | 0.916 | 0.084 (0.049, 0.119) | 0.047 | 0.762 |
| Harvest (7) | strips R=4 | 1 | 7 migrator | handover | 0.159 | 0.139 | 0.021 (-0.068, 0.110) | 0.909 | 0.037 |
| Harvest (7) | strips R=4 | 1 | 7 migrator | local | 0.159 | 0.139 | 0.021 (-0.068, 0.110) | 0.909 | 0.037 |
| Harvest (7) | strips R=4 | 10 | 7 migrator | handover | 0.984 | 0.837 | 0.147 (0.102, 0.196) | 0.890 | 0.469 |
| Harvest (7) | strips R=4 | 10 | 7 migrator | local | 0.984 | 0.781 | 0.202 (0.157, 0.253) | 0.896 | 0.469 |
| Harvest (7) | strips R=4 | 50 | 7 migrator | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.881 | 0.817 |
| Harvest (7) | strips R=4 | 50 | 7 migrator | local | 1.000 | 0.936 | 0.064 (0.016, 0.136) | 0.885 | 0.817 |
| Harvest (7) | strips R=4 | 1 | 7 sneaky | handover | 0.159 | 0.162 | -0.003 (-0.107, 0.104) | 0.571 | 0.031 |
| Harvest (7) | strips R=4 | 1 | 7 sneaky | local | 0.159 | 0.162 | -0.003 (-0.107, 0.104) | 0.571 | 0.031 |
| Harvest (7) | strips R=4 | 10 | 7 sneaky | handover | 0.984 | 0.867 | 0.117 (0.065, 0.161) | 0.513 | 0.426 |
| Harvest (7) | strips R=4 | 10 | 7 sneaky | local | 0.984 | 0.821 | 0.163 (0.127, 0.201) | 0.467 | 0.426 |
| Harvest (7) | strips R=4 | 50 | 7 sneaky | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.513 | 0.924 |
| Harvest (7) | strips R=4 | 50 | 7 sneaky | local | 1.000 | 0.919 | 0.081 (0.049, 0.110) | 0.468 | 0.924 |
| Harvest, large (14) | strips R=2 | 1 | 1 greedy + 13 honest | handover | 0.269 | 0.393 | -0.124 (-0.234, -0.029) | -0.135 | 0.024 |
| Harvest, large (14) | strips R=2 | 1 | 1 greedy + 13 honest | local | 0.269 | 0.393 | -0.124 (-0.234, -0.029) | -0.135 | 0.024 |
| Harvest, large (14) | strips R=2 | 10 | 1 greedy + 13 honest | handover | 0.962 | 0.975 | -0.013 (-0.035, 0.005) | -0.069 | 0.133 |
| Harvest, large (14) | strips R=2 | 10 | 1 greedy + 13 honest | local | 0.962 | 0.956 | 0.006 (-0.033, 0.044) | -0.069 | 0.133 |
| Harvest, large (14) | strips R=2 | 50 | 1 greedy + 13 honest | handover | 0.995 | 0.994 | 0.001 (-0.015, 0.016) | -0.069 | 0.450 |
| Harvest, large (14) | strips R=2 | 50 | 1 greedy + 13 honest | local | 0.995 | 0.975 | 0.020 (0.001, 0.042) | -0.069 | 0.450 |
| Harvest, large (14) | strips R=2 | 1 | 14 migrator | handover | 0.333 | 0.193 | 0.141 (0.069, 0.214) | 0.470 | 0.027 |
| Harvest, large (14) | strips R=2 | 1 | 14 migrator | local | 0.333 | 0.193 | 0.141 (0.069, 0.214) | 0.470 | 0.027 |
| Harvest, large (14) | strips R=2 | 10 | 14 migrator | handover | 0.979 | 0.891 | 0.088 (0.010, 0.151) | 0.452 | 0.106 |
| Harvest, large (14) | strips R=2 | 10 | 14 migrator | local | 0.979 | 0.887 | 0.092 (0.010, 0.152) | 0.452 | 0.106 |
| Harvest, large (14) | strips R=2 | 50 | 14 migrator | handover | 1.000 | 0.996 | 0.004 (0.000, 0.014) | 0.463 | 0.663 |
| Harvest, large (14) | strips R=2 | 50 | 14 migrator | local | 1.000 | 0.992 | 0.008 (0.000, 0.017) | 0.463 | 0.663 |
| Harvest, large (14) | strips R=2 | 1 | 14 sneaky | handover | 0.333 | 0.232 | 0.101 (-0.005, 0.194) | 0.088 | 0.044 |
| Harvest, large (14) | strips R=2 | 1 | 14 sneaky | local | 0.333 | 0.232 | 0.101 (-0.005, 0.194) | 0.088 | 0.044 |
| Harvest, large (14) | strips R=2 | 10 | 14 sneaky | handover | 0.979 | 0.949 | 0.030 (0.000, 0.084) | 0.072 | 0.135 |
| Harvest, large (14) | strips R=2 | 10 | 14 sneaky | local | 0.979 | 0.880 | 0.099 (0.009, 0.195) | 0.063 | 0.135 |
| Harvest, large (14) | strips R=2 | 50 | 14 sneaky | handover | 1.000 | 0.983 | 0.017 (0.000, 0.056) | 0.060 | 0.399 |
| Harvest, large (14) | strips R=2 | 50 | 14 sneaky | local | 1.000 | 0.910 | 0.090 (0.009, 0.216) | 0.051 | 0.399 |
| Harvest, large (14) | strips R=4 | 1 | 1 greedy + 13 honest | handover | 0.269 | 0.413 | -0.145 (-0.243, -0.083) | -0.284 | 0.024 |
| Harvest, large (14) | strips R=4 | 1 | 1 greedy + 13 honest | local | 0.269 | 0.413 | -0.145 (-0.243, -0.083) | -0.284 | 0.024 |
| Harvest, large (14) | strips R=4 | 10 | 1 greedy + 13 honest | handover | 0.962 | 0.908 | 0.054 (0.008, 0.125) | -0.070 | 0.170 |
| Harvest, large (14) | strips R=4 | 10 | 1 greedy + 13 honest | local | 0.962 | 0.887 | 0.074 (0.019, 0.128) | -0.074 | 0.170 |
| Harvest, large (14) | strips R=4 | 50 | 1 greedy + 13 honest | handover | 0.995 | 0.994 | 0.000 (-0.016, 0.021) | -0.063 | 0.781 |
| Harvest, large (14) | strips R=4 | 50 | 1 greedy + 13 honest | local | 0.995 | 0.975 | 0.019 (0.007, 0.032) | -0.065 | 0.781 |
| Harvest, large (14) | strips R=4 | 1 | 14 migrator | handover | 0.333 | 0.142 | 0.191 (0.127, 0.257) | 0.828 | 0.018 |
| Harvest, large (14) | strips R=4 | 1 | 14 migrator | local | 0.333 | 0.142 | 0.191 (0.127, 0.257) | 0.828 | 0.018 |
| Harvest, large (14) | strips R=4 | 10 | 14 migrator | handover | 0.979 | 0.795 | 0.184 (0.120, 0.260) | 0.822 | 0.167 |
| Harvest, large (14) | strips R=4 | 10 | 14 migrator | local | 0.979 | 0.802 | 0.177 (0.108, 0.260) | 0.821 | 0.167 |
| Harvest, large (14) | strips R=4 | 50 | 14 migrator | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.816 | 0.909 |
| Harvest, large (14) | strips R=4 | 50 | 14 migrator | local | 1.000 | 0.983 | 0.017 (0.000, 0.040) | 0.812 | 0.909 |
| Harvest, large (14) | strips R=4 | 1 | 14 sneaky | handover | 0.333 | 0.136 | 0.197 (0.152, 0.243) | 0.453 | 0.035 |
| Harvest, large (14) | strips R=4 | 1 | 14 sneaky | local | 0.333 | 0.136 | 0.197 (0.152, 0.243) | 0.453 | 0.035 |
| Harvest, large (14) | strips R=4 | 10 | 14 sneaky | handover | 0.979 | 0.959 | 0.020 (-0.056, 0.079) | 0.323 | 0.354 |
| Harvest, large (14) | strips R=4 | 10 | 14 sneaky | local | 0.979 | 0.829 | 0.149 (0.089, 0.221) | 0.349 | 0.354 |
| Harvest, large (14) | strips R=4 | 50 | 14 sneaky | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.319 | 0.833 |
| Harvest, large (14) | strips R=4 | 50 | 14 sneaky | local | 1.000 | 0.940 | 0.060 (0.025, 0.090) | 0.323 | 0.833 |
| Rooms, closed (7) | rooms (R=2) | 1 | 1 greedy + 6 honest | handover | 0.246 | 0.335 | -0.089 (-0.181, -0.002) | -0.096 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 1 | 1 greedy + 6 honest | local | 0.246 | 0.335 | -0.089 (-0.181, -0.002) | -0.096 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 10 | 1 greedy + 6 honest | handover | 0.965 | 0.951 | 0.014 (-0.041, 0.066) | -0.096 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 10 | 1 greedy + 6 honest | local | 0.965 | 0.951 | 0.014 (-0.041, 0.066) | -0.096 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 50 | 1 greedy + 6 honest | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | -0.095 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 50 | 1 greedy + 6 honest | local | 1.000 | 1.000 | 0.000 (0.000, 0.000) | -0.095 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 1 | 7 migrator | handover | 0.246 | 0.220 | 0.026 (-0.091, 0.141) | 0.109 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 1 | 7 migrator | local | 0.246 | 0.220 | 0.026 (-0.091, 0.141) | 0.109 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 10 | 7 migrator | handover | 0.986 | 0.877 | 0.109 (0.041, 0.187) | 0.092 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 10 | 7 migrator | local | 0.986 | 0.877 | 0.109 (0.041, 0.187) | 0.092 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 50 | 7 migrator | handover | 1.000 | 0.965 | 0.035 (0.000, 0.126) | 0.093 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 50 | 7 migrator | local | 1.000 | 0.965 | 0.035 (0.000, 0.126) | 0.093 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 1 | 7 sneaky | handover | 0.246 | 0.220 | 0.026 (-0.091, 0.141) | 0.109 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 1 | 7 sneaky | local | 0.246 | 0.220 | 0.026 (-0.091, 0.141) | 0.109 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 10 | 7 sneaky | handover | 0.986 | 0.877 | 0.109 (0.041, 0.187) | 0.092 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 10 | 7 sneaky | local | 0.986 | 0.877 | 0.109 (0.041, 0.187) | 0.092 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 50 | 7 sneaky | handover | 1.000 | 0.965 | 0.035 (0.000, 0.126) | 0.093 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 50 | 7 sneaky | local | 1.000 | 0.965 | 0.035 (0.000, 0.126) | 0.093 | 0.000 |
| Rooms, closed (7) | strips R=4 | 1 | 1 greedy + 6 honest | handover | 0.246 | 0.271 | -0.025 (-0.095, 0.030) | -0.028 | 0.028 |
| Rooms, closed (7) | strips R=4 | 1 | 1 greedy + 6 honest | local | 0.246 | 0.271 | -0.025 (-0.095, 0.030) | -0.028 | 0.028 |
| Rooms, closed (7) | strips R=4 | 10 | 1 greedy + 6 honest | handover | 0.965 | 0.924 | 0.041 (-0.010, 0.098) | 0.015 | 0.242 |
| Rooms, closed (7) | strips R=4 | 10 | 1 greedy + 6 honest | local | 0.965 | 0.876 | 0.089 (0.026, 0.164) | 0.032 | 0.242 |
| Rooms, closed (7) | strips R=4 | 50 | 1 greedy + 6 honest | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.023 | 0.723 |
| Rooms, closed (7) | strips R=4 | 50 | 1 greedy + 6 honest | local | 1.000 | 0.982 | 0.018 (0.000, 0.038) | 0.041 | 0.723 |
| Rooms, closed (7) | strips R=4 | 1 | 7 migrator | handover | 0.246 | 0.229 | 0.017 (-0.062, 0.121) | 0.823 | 0.033 |
| Rooms, closed (7) | strips R=4 | 1 | 7 migrator | local | 0.246 | 0.229 | 0.017 (-0.062, 0.121) | 0.823 | 0.033 |
| Rooms, closed (7) | strips R=4 | 10 | 7 migrator | handover | 0.986 | 0.921 | 0.065 (0.022, 0.102) | 0.810 | 0.194 |
| Rooms, closed (7) | strips R=4 | 10 | 7 migrator | local | 0.986 | 0.914 | 0.071 (0.027, 0.112) | 0.810 | 0.194 |
| Rooms, closed (7) | strips R=4 | 50 | 7 migrator | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.814 | 0.763 |
| Rooms, closed (7) | strips R=4 | 50 | 7 migrator | local | 1.000 | 0.987 | 0.013 (0.003, 0.022) | 0.814 | 0.763 |
| Rooms, closed (7) | strips R=4 | 1 | 7 sneaky | handover | 0.246 | 0.181 | 0.065 (-0.007, 0.142) | 0.237 | 0.029 |
| Rooms, closed (7) | strips R=4 | 1 | 7 sneaky | local | 0.246 | 0.181 | 0.065 (-0.007, 0.142) | 0.237 | 0.029 |
| Rooms, closed (7) | strips R=4 | 10 | 7 sneaky | handover | 0.986 | 0.861 | 0.125 (0.073, 0.180) | 0.273 | 0.190 |
| Rooms, closed (7) | strips R=4 | 10 | 7 sneaky | local | 0.986 | 0.846 | 0.140 (0.088, 0.184) | 0.284 | 0.190 |
| Rooms, closed (7) | strips R=4 | 50 | 7 sneaky | handover | 1.000 | 0.980 | 0.020 (0.000, 0.065) | 0.246 | 0.742 |
| Rooms, closed (7) | strips R=4 | 50 | 7 sneaky | local | 1.000 | 0.945 | 0.055 (0.009, 0.113) | 0.265 | 0.742 |
| Rooms, one door (7) | rooms (R=2) | 1 | 1 greedy + 6 honest | handover | 0.270 | 0.357 | -0.088 (-0.164, -0.011) | -0.173 | 0.000 |
| Rooms, one door (7) | rooms (R=2) | 1 | 1 greedy + 6 honest | local | 0.270 | 0.357 | -0.088 (-0.164, -0.011) | -0.173 | 0.000 |
| Rooms, one door (7) | rooms (R=2) | 10 | 1 greedy + 6 honest | handover | 0.983 | 0.934 | 0.049 (0.023, 0.078) | -0.110 | 0.000 |
| Rooms, one door (7) | rooms (R=2) | 10 | 1 greedy + 6 honest | local | 0.983 | 0.934 | 0.049 (0.023, 0.078) | -0.110 | 0.000 |
| Rooms, one door (7) | rooms (R=2) | 50 | 1 greedy + 6 honest | handover | 1.000 | 0.986 | 0.014 (0.000, 0.043) | -0.126 | 0.324 |
| Rooms, one door (7) | rooms (R=2) | 50 | 1 greedy + 6 honest | local | 1.000 | 0.986 | 0.014 (0.000, 0.043) | -0.126 | 0.324 |
| Rooms, one door (7) | rooms (R=2) | 1 | 7 migrator | handover | 0.180 | 0.211 | -0.030 (-0.121, 0.040) | 0.351 | 0.000 |
| Rooms, one door (7) | rooms (R=2) | 1 | 7 migrator | local | 0.180 | 0.211 | -0.030 (-0.121, 0.040) | 0.351 | 0.000 |
| Rooms, one door (7) | rooms (R=2) | 10 | 7 migrator | handover | 0.960 | 0.904 | 0.056 (-0.013, 0.102) | 0.265 | 0.000 |
| Rooms, one door (7) | rooms (R=2) | 10 | 7 migrator | local | 0.960 | 0.904 | 0.056 (-0.013, 0.102) | 0.265 | 0.000 |
| Rooms, one door (7) | rooms (R=2) | 50 | 7 migrator | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.259 | 0.573 |
| Rooms, one door (7) | rooms (R=2) | 50 | 7 migrator | local | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.259 | 0.573 |
| Rooms, one door (7) | rooms (R=2) | 1 | 7 sneaky | handover | 0.180 | 0.171 | 0.010 (-0.083, 0.083) | 0.205 | 0.000 |
| Rooms, one door (7) | rooms (R=2) | 1 | 7 sneaky | local | 0.180 | 0.171 | 0.010 (-0.083, 0.083) | 0.205 | 0.000 |
| Rooms, one door (7) | rooms (R=2) | 10 | 7 sneaky | handover | 0.960 | 0.910 | 0.050 (-0.026, 0.121) | 0.086 | 0.000 |
| Rooms, one door (7) | rooms (R=2) | 10 | 7 sneaky | local | 0.960 | 0.910 | 0.050 (-0.026, 0.121) | 0.086 | 0.000 |
| Rooms, one door (7) | rooms (R=2) | 50 | 7 sneaky | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.090 | 0.359 |
| Rooms, one door (7) | rooms (R=2) | 50 | 7 sneaky | local | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.090 | 0.359 |
| Rooms, large (14) | rooms (R=4) | 1 | 1 greedy + 13 honest | handover | 0.289 | 0.292 | -0.003 (-0.079, 0.091) | -0.022 | 0.010 |
| Rooms, large (14) | rooms (R=4) | 1 | 1 greedy + 13 honest | local | 0.289 | 0.292 | -0.003 (-0.079, 0.091) | -0.022 | 0.010 |
| Rooms, large (14) | rooms (R=4) | 10 | 1 greedy + 13 honest | handover | 0.982 | 0.974 | 0.008 (-0.024, 0.036) | -0.178 | 0.010 |
| Rooms, large (14) | rooms (R=4) | 10 | 1 greedy + 13 honest | local | 0.982 | 0.974 | 0.008 (-0.024, 0.036) | -0.178 | 0.010 |
| Rooms, large (14) | rooms (R=4) | 50 | 1 greedy + 13 honest | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | -0.194 | 0.010 |
| Rooms, large (14) | rooms (R=4) | 50 | 1 greedy + 13 honest | local | 1.000 | 1.000 | 0.000 (0.000, 0.000) | -0.194 | 0.010 |
| Rooms, large (14) | rooms (R=4) | 1 | 14 migrator | handover | 0.224 | 0.200 | 0.024 (-0.083, 0.158) | 0.152 | 0.000 |
| Rooms, large (14) | rooms (R=4) | 1 | 14 migrator | local | 0.224 | 0.200 | 0.024 (-0.083, 0.158) | 0.152 | 0.000 |
| Rooms, large (14) | rooms (R=4) | 10 | 14 migrator | handover | 1.000 | 0.905 | 0.095 (0.037, 0.162) | 0.150 | 0.000 |
| Rooms, large (14) | rooms (R=4) | 10 | 14 migrator | local | 1.000 | 0.905 | 0.095 (0.037, 0.162) | 0.150 | 0.000 |
| Rooms, large (14) | rooms (R=4) | 50 | 14 migrator | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.155 | 0.000 |
| Rooms, large (14) | rooms (R=4) | 50 | 14 migrator | local | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.155 | 0.000 |
| Rooms, large (14) | rooms (R=4) | 1 | 14 sneaky | handover | 0.224 | 0.200 | 0.024 (-0.083, 0.158) | 0.152 | 0.000 |
| Rooms, large (14) | rooms (R=4) | 1 | 14 sneaky | local | 0.224 | 0.200 | 0.024 (-0.083, 0.158) | 0.152 | 0.000 |
| Rooms, large (14) | rooms (R=4) | 10 | 14 sneaky | handover | 1.000 | 0.905 | 0.095 (0.037, 0.162) | 0.150 | 0.000 |
| Rooms, large (14) | rooms (R=4) | 10 | 14 sneaky | local | 1.000 | 0.905 | 0.095 (0.037, 0.162) | 0.150 | 0.000 |
| Rooms, large (14) | rooms (R=4) | 50 | 14 sneaky | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.155 | 0.000 |
| Rooms, large (14) | rooms (R=4) | 50 | 14 sneaky | local | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.155 | 0.000 |
| Rooms, large (14) | strips R=2 | 1 | 1 greedy + 13 honest | handover | 0.289 | 0.324 | -0.035 (-0.151, 0.031) | -0.003 | 0.010 |
| Rooms, large (14) | strips R=2 | 1 | 1 greedy + 13 honest | local | 0.289 | 0.324 | -0.035 (-0.151, 0.031) | -0.003 | 0.010 |
| Rooms, large (14) | strips R=2 | 10 | 1 greedy + 13 honest | handover | 0.982 | 0.991 | -0.008 (-0.025, 0.000) | -0.091 | 0.010 |
| Rooms, large (14) | strips R=2 | 10 | 1 greedy + 13 honest | local | 0.982 | 0.991 | -0.008 (-0.025, 0.000) | -0.091 | 0.010 |
| Rooms, large (14) | strips R=2 | 50 | 1 greedy + 13 honest | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | -0.091 | 0.010 |
| Rooms, large (14) | strips R=2 | 50 | 1 greedy + 13 honest | local | 1.000 | 1.000 | 0.000 (0.000, 0.000) | -0.091 | 0.010 |
| Rooms, large (14) | strips R=2 | 1 | 14 migrator | handover | 0.224 | 0.149 | 0.076 (-0.038, 0.204) | 0.048 | 0.019 |
| Rooms, large (14) | strips R=2 | 1 | 14 migrator | local | 0.224 | 0.149 | 0.076 (-0.038, 0.204) | 0.048 | 0.019 |
| Rooms, large (14) | strips R=2 | 10 | 14 migrator | handover | 1.000 | 0.923 | 0.077 (0.020, 0.137) | 0.079 | 0.020 |
| Rooms, large (14) | strips R=2 | 10 | 14 migrator | local | 1.000 | 0.923 | 0.077 (0.020, 0.137) | 0.079 | 0.020 |
| Rooms, large (14) | strips R=2 | 50 | 14 migrator | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.080 | 0.020 |
| Rooms, large (14) | strips R=2 | 50 | 14 migrator | local | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.080 | 0.020 |
| Rooms, large (14) | strips R=2 | 1 | 14 sneaky | handover | 0.224 | 0.149 | 0.076 (-0.038, 0.204) | 0.048 | 0.019 |
| Rooms, large (14) | strips R=2 | 1 | 14 sneaky | local | 0.224 | 0.149 | 0.076 (-0.038, 0.204) | 0.048 | 0.019 |
| Rooms, large (14) | strips R=2 | 10 | 14 sneaky | handover | 1.000 | 0.923 | 0.077 (0.020, 0.137) | 0.079 | 0.020 |
| Rooms, large (14) | strips R=2 | 10 | 14 sneaky | local | 1.000 | 0.923 | 0.077 (0.020, 0.137) | 0.079 | 0.020 |
| Rooms, large (14) | strips R=2 | 50 | 14 sneaky | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.080 | 0.020 |
| Rooms, large (14) | strips R=2 | 50 | 14 sneaky | local | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.080 | 0.020 |
| Coins (2) | bands R=2 | 1 | 1 greedy + 1 honest | handover | 0.462 | 0.371 | 0.090 (0.057, 0.125) | 0.068 | 0.320 |
| Coins (2) | bands R=2 | 1 | 1 greedy + 1 honest | local | 0.462 | 0.371 | 0.090 (0.057, 0.125) | 0.068 | 0.320 |
| Coins (2) | bands R=2 | 10 | 1 greedy + 1 honest | handover | 0.995 | 0.968 | 0.028 (0.000, 0.057) | 0.064 | 0.543 |
| Coins (2) | bands R=2 | 10 | 1 greedy + 1 honest | local | 0.995 | 0.882 | 0.113 (0.070, 0.156) | 0.104 | 0.543 |
| Coins (2) | bands R=2 | 50 | 1 greedy + 1 honest | handover | 0.995 | 0.986 | 0.010 (-0.014, 0.039) | 0.024 | 0.974 |
| Coins (2) | bands R=2 | 50 | 1 greedy + 1 honest | local | 0.995 | 0.963 | 0.032 (0.010, 0.054) | 0.068 | 0.974 |
| Coins (2) | bands R=2 | 1 | 2 migrator | handover | 0.625 | 0.262 | 0.363 (-0.024, 0.696) | 0.041 | 0.375 |
| Coins (2) | bands R=2 | 1 | 2 migrator | local | 0.625 | 0.262 | 0.363 (-0.024, 0.696) | 0.041 | 0.375 |
| Coins (2) | bands R=2 | 10 | 2 migrator | handover | 1.000 | 0.985 | 0.015 (0.000, 0.042) | 0.077 | 0.660 |
| Coins (2) | bands R=2 | 10 | 2 migrator | local | 1.000 | 0.942 | 0.058 (0.015, 0.111) | 0.080 | 0.660 |
| Coins (2) | bands R=2 | 50 | 2 migrator | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.077 | 0.977 |
| Coins (2) | bands R=2 | 50 | 2 migrator | local | 1.000 | 0.986 | 0.014 (0.000, 0.041) | 0.080 | 0.977 |
| Coins (2) | bands R=2 | 1 | 2 sneaky | handover | 0.625 | 0.375 | 0.250 (0.000, 0.556) | 0.000 | 0.571 |
| Coins (2) | bands R=2 | 1 | 2 sneaky | local | 0.625 | 0.375 | 0.250 (0.000, 0.556) | 0.000 | 0.571 |
| Coins (2) | bands R=2 | 10 | 2 sneaky | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.000 | 0.714 |
| Coins (2) | bands R=2 | 10 | 2 sneaky | local | 1.000 | 0.875 | 0.125 (0.000, 0.500) | 0.000 | 0.714 |
| Coins (2) | bands R=2 | 50 | 2 sneaky | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.000 | 1.000 |
| Coins (2) | bands R=2 | 50 | 2 sneaky | local | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.001 | 1.000 |
| Clean Up (7) | bands R=2 | 150 | 1 greedy + 6 honest | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | -0.409 | 0.937 |
| Clean Up (7) | bands R=2 | 150 | 1 greedy + 6 honest | local | 1.000 | 1.000 | 0.000 (0.000, 0.000) | -0.409 | 0.937 |
| Clean Up (7) | bands R=2 | 150 | 7 migrator | handover | – | – | – (–, –) | 0.153 | 0.999 |
| Clean Up (7) | bands R=2 | 150 | 7 migrator | local | – | – | – (–, –) | 0.153 | 0.999 |
| Clean Up (7) | bands R=2 | 150 | 7 sneaky | handover | – | – | – (–, –) | 0.032 | 0.962 |
| Clean Up (7) | bands R=2 | 150 | 7 sneaky | local | – | – | – (–, –) | 0.032 | 0.962 |
| Clean Up (7) | strips R=2 | 150 | 1 greedy + 6 honest | handover | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.120 | 0.868 |
| Clean Up (7) | strips R=2 | 150 | 1 greedy + 6 honest | local | 1.000 | 0.965 | 0.035 (0.000, 0.082) | -0.302 | 0.868 |
| Clean Up (7) | strips R=2 | 150 | 7 migrator | handover | – | – | – (–, –) | 0.170 | 0.966 |
| Clean Up (7) | strips R=2 | 150 | 7 migrator | local | – | – | – (–, –) | 0.170 | 0.966 |
| Clean Up (7) | strips R=2 | 150 | 7 sneaky | handover | – | – | – (–, –) | -0.040 | 0.884 |
| Clean Up (7) | strips R=2 | 150 | 7 sneaky | local | – | – | – (–, –) | -0.040 | 0.884 |

### Does ξ_loc order the gap? (Spearman ρ, points with ≥ 20 violations under both overseers)

| Records | Populations | predictor | ρ | points |
|---|---|---|---|---|
| local | 1 greedy, all sneaky | ξ_loc = 1 − Λ_w | 0.42 | 65 |
| local | 1 greedy, all sneaky | 1 − S_w (evidence leaves, no ζ) | 0.33 | 65 |
| local | all migrator | ξ_loc = 1 − Λ_w | 0.02 | 30 |
| local | all migrator | 1 − S_w (evidence leaves, no ζ) | 0.02 | 30 |
| handover | 1 greedy, all sneaky | ξ_loc = 1 − Λ_w | 0.08 | 65 |
| handover | 1 greedy, all sneaky | 1 − S_w (evidence leaves, no ζ) | -0.02 | 65 |
| handover | all migrator | ξ_loc = 1 − Λ_w | -0.14 | 30 |
| handover | all migrator | 1 − S_w (evidence leaves, no ζ) | -0.14 | 30 |

### The part of the gap that comes from records staying local (regional with hand-over − regional with local records), against what predicts it

| Populations | predictor | ρ | points |
|---|---|---|---|
| 1 greedy, all sneaky | ξ_loc = 1 − Λ_w | 0.80 | 65 |
| 1 greedy, all sneaky | 1 − S_w | 0.85 | 65 |
| 1 greedy, all sneaky | P_C − P_0 (formula, q = K/N, measured χ) | 0.76 | 65 |
| all migrator | ξ_loc = 1 − Λ_w | 0.69 | 33 |
| all migrator | 1 − S_w | 0.79 | 33 |
| all migrator | P_C − P_0 (formula, q = K/N, measured χ) | 0.60 | 33 |

| Testbed | Regions | w | Population | χ | ξ_loc | P_C − P_0 | lost by local records |
|---|---|---|---|---|---|---|---|
| Harvest (7) | grid R=4 | 1 | 1 greedy + 6 honest | 0.0452 | 0.101 | 0.000 | 0.000 |
| Harvest (7) | grid R=4 | 10 | 1 greedy + 6 honest | 0.0452 | 0.374 | 0.087 | 0.096 |
| Harvest (7) | grid R=4 | 50 | 1 greedy + 6 honest | 0.0452 | 0.922 | 0.102 | 0.145 |
| Harvest (7) | grid R=4 | 1 | 7 sneaky | 0.0543 | 0.118 | 0.000 | 0.000 |
| Harvest (7) | grid R=4 | 10 | 7 sneaky | 0.0543 | 0.461 | 0.102 | 0.151 |
| Harvest (7) | grid R=4 | 50 | 7 sneaky | 0.0543 | 0.878 | 0.120 | 0.160 |
| Harvest (7) | strips R=2 | 1 | 1 greedy + 6 honest | 0.0179 | 0.000 | 0.000 | 0.000 |
| Harvest (7) | strips R=2 | 10 | 1 greedy + 6 honest | 0.0179 | 0.032 | 0.036 | 0.001 |
| Harvest (7) | strips R=2 | 50 | 1 greedy + 6 honest | 0.0179 | 0.455 | 0.043 | 0.024 |
| Harvest (7) | strips R=2 | 1 | 7 sneaky | 0.0280 | 0.000 | 0.000 | 0.000 |
| Harvest (7) | strips R=2 | 10 | 7 sneaky | 0.0280 | 0.263 | 0.055 | 0.000 |
| Harvest (7) | strips R=2 | 50 | 7 sneaky | 0.0280 | 0.753 | 0.065 | 0.000 |
| Harvest (7) | strips R=4 | 1 | 1 greedy + 6 honest | 0.0625 | 0.035 | 0.000 | 0.000 |
| Harvest (7) | strips R=4 | 10 | 1 greedy + 6 honest | 0.0625 | 0.366 | 0.116 | 0.021 |
| Harvest (7) | strips R=4 | 50 | 1 greedy + 6 honest | 0.0625 | 0.762 | 0.135 | 0.084 |
| Harvest (7) | strips R=4 | 1 | 7 sneaky | 0.0806 | 0.031 | 0.000 | 0.000 |
| Harvest (7) | strips R=4 | 10 | 7 sneaky | 0.0806 | 0.426 | 0.146 | 0.046 |
| Harvest (7) | strips R=4 | 50 | 7 sneaky | 0.0806 | 0.924 | 0.168 | 0.081 |
| Harvest, large (14) | strips R=2 | 1 | 1 greedy + 13 honest | 0.0085 | 0.024 | 0.000 | 0.000 |
| Harvest, large (14) | strips R=2 | 10 | 1 greedy + 13 honest | 0.0085 | 0.133 | 0.017 | 0.019 |
| Harvest, large (14) | strips R=2 | 50 | 1 greedy + 13 honest | 0.0085 | 0.450 | 0.021 | 0.019 |
| Harvest, large (14) | strips R=2 | 1 | 14 sneaky | 0.0165 | 0.044 | 0.000 | 0.000 |
| Harvest, large (14) | strips R=2 | 10 | 14 sneaky | 0.0165 | 0.135 | 0.033 | 0.069 |
| Harvest, large (14) | strips R=2 | 50 | 14 sneaky | 0.0165 | 0.399 | 0.040 | 0.073 |
| Harvest, large (14) | strips R=4 | 1 | 1 greedy + 13 honest | 0.0296 | 0.024 | 0.000 | 0.000 |
| Harvest, large (14) | strips R=4 | 10 | 1 greedy + 13 honest | 0.0296 | 0.170 | 0.058 | 0.020 |
| Harvest, large (14) | strips R=4 | 50 | 1 greedy + 13 honest | 0.0296 | 0.781 | 0.069 | 0.019 |
| Harvest, large (14) | strips R=4 | 1 | 14 sneaky | 0.0455 | 0.035 | 0.000 | 0.000 |
| Harvest, large (14) | strips R=4 | 10 | 14 sneaky | 0.0455 | 0.354 | 0.087 | 0.129 |
| Harvest, large (14) | strips R=4 | 50 | 14 sneaky | 0.0455 | 0.833 | 0.102 | 0.060 |
| Rooms, closed (7) | rooms (R=2) | 1 | 1 greedy + 6 honest | 0.0000 | 0.000 | 0.000 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 10 | 1 greedy + 6 honest | 0.0000 | 0.000 | 0.000 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 50 | 1 greedy + 6 honest | 0.0000 | 0.000 | -0.000 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 1 | 7 sneaky | 0.0000 | 0.000 | 0.000 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 10 | 7 sneaky | 0.0000 | 0.000 | 0.000 | 0.000 |
| Rooms, closed (7) | rooms (R=2) | 50 | 7 sneaky | 0.0000 | 0.000 | -0.000 | 0.000 |
| Rooms, closed (7) | strips R=4 | 1 | 1 greedy + 6 honest | 0.0411 | 0.028 | 0.000 | 0.000 |
| Rooms, closed (7) | strips R=4 | 10 | 1 greedy + 6 honest | 0.0411 | 0.242 | 0.079 | 0.048 |
| Rooms, closed (7) | strips R=4 | 50 | 1 greedy + 6 honest | 0.0411 | 0.723 | 0.093 | 0.018 |
| Rooms, closed (7) | strips R=4 | 1 | 7 sneaky | 0.0429 | 0.029 | 0.000 | 0.000 |
| Rooms, closed (7) | strips R=4 | 10 | 7 sneaky | 0.0429 | 0.190 | 0.082 | 0.015 |
| Rooms, closed (7) | strips R=4 | 50 | 7 sneaky | 0.0429 | 0.742 | 0.097 | 0.034 |
| Rooms, one door (7) | rooms (R=2) | 1 | 1 greedy + 6 honest | 0.0019 | 0.000 | 0.000 | 0.000 |
| Rooms, one door (7) | rooms (R=2) | 10 | 1 greedy + 6 honest | 0.0019 | 0.000 | 0.004 | 0.000 |
| Rooms, one door (7) | rooms (R=2) | 50 | 1 greedy + 6 honest | 0.0019 | 0.324 | 0.005 | 0.000 |
| Rooms, one door (7) | rooms (R=2) | 1 | 7 sneaky | 0.0062 | 0.000 | 0.000 | 0.000 |
| Rooms, one door (7) | rooms (R=2) | 10 | 7 sneaky | 0.0062 | 0.000 | 0.013 | 0.000 |
| Rooms, one door (7) | rooms (R=2) | 50 | 7 sneaky | 0.0062 | 0.359 | 0.015 | 0.000 |
| Rooms, large (14) | rooms (R=4) | 1 | 1 greedy + 13 honest | 0.0000 | 0.010 | 0.000 | 0.000 |
| Rooms, large (14) | rooms (R=4) | 10 | 1 greedy + 13 honest | 0.0000 | 0.010 | 0.000 | 0.000 |
| Rooms, large (14) | rooms (R=4) | 50 | 1 greedy + 13 honest | 0.0000 | 0.010 | -0.000 | 0.000 |
| Rooms, large (14) | rooms (R=4) | 1 | 14 sneaky | 0.0000 | 0.000 | 0.000 | 0.000 |
| Rooms, large (14) | rooms (R=4) | 10 | 14 sneaky | 0.0000 | 0.000 | 0.000 | 0.000 |
| Rooms, large (14) | rooms (R=4) | 50 | 14 sneaky | 0.0000 | 0.000 | -0.000 | 0.000 |
| Rooms, large (14) | strips R=2 | 1 | 1 greedy + 13 honest | 0.0000 | 0.010 | 0.000 | 0.000 |
| Rooms, large (14) | strips R=2 | 10 | 1 greedy + 13 honest | 0.0000 | 0.010 | 0.000 | 0.000 |
| Rooms, large (14) | strips R=2 | 50 | 1 greedy + 13 honest | 0.0000 | 0.010 | -0.000 | 0.000 |
| Rooms, large (14) | strips R=2 | 1 | 14 sneaky | 0.0000 | 0.019 | 0.000 | 0.000 |
| Rooms, large (14) | strips R=2 | 10 | 14 sneaky | 0.0000 | 0.020 | 0.000 | 0.000 |
| Rooms, large (14) | strips R=2 | 50 | 14 sneaky | 0.0000 | 0.020 | -0.000 | 0.000 |
| Coins (2) | bands R=2 | 1 | 1 greedy + 1 honest | 0.0471 | 0.320 | 0.000 | 0.000 |
| Coins (2) | bands R=2 | 10 | 1 greedy + 1 honest | 0.0471 | 0.543 | 0.045 | 0.086 |
| Coins (2) | bands R=2 | 50 | 1 greedy + 1 honest | 0.0471 | 0.974 | 0.045 | 0.022 |
| Clean Up (7) | bands R=2 | 150 | 1 greedy + 6 honest | 0.0184 | 0.937 | 0.044 | 0.000 |
| Clean Up (7) | strips R=2 | 150 | 1 greedy + 6 honest | 0.0090 | 0.868 | 0.022 | 0.035 |

### Task 4: Willis et al.'s commons, K = N/8 audits per round, 50 games per N

| N | Chooser | Groups formed | M | Budget | central | groups | Δ_dec (± 1.96 SE) | numbers between overseers per round, % of central's N records |
|---|---|---|---|---|---|---|---|---|
| 64 | history | by attitude | 4 | fixed K/M | 0.164 | 0.155 | +0.009 (± 0.008) | 0.0% |
| 64 | history | by attitude | 4 | pooled | 0.164 | 0.157 | +0.007 (± 0.009) | 12.5% |
| 64 | history | by attitude | 16 | fixed K/M | 0.164 | 0.162 | +0.002 (± 0.015) | 0.0% |
| 64 | history | by attitude | 16 | pooled | 0.164 | 0.164 | -0.000 (± 0.010) | 50.0% |
| 64 | history | random | 4 | fixed K/M | 0.164 | 0.164 | +0.000 (± 0.009) | 0.0% |
| 64 | history | random | 4 | pooled | 0.164 | 0.162 | +0.003 (± 0.009) | 12.5% |
| 64 | history | random | 16 | fixed K/M | 0.164 | 0.161 | +0.003 (± 0.012) | 0.0% |
| 64 | history | random | 16 | pooled | 0.164 | 0.164 | +0.000 (± 0.010) | 50.0% |
| 64 | random | by attitude | 4 | fixed K/M | 0.124 | 0.124 | +0.001 (± 0.003) | 0.0% |
| 64 | random | by attitude | 4 | pooled | 0.124 | 0.142 | -0.018 (± 0.007) | 12.5% |
| 64 | random | by attitude | 16 | fixed K/M | 0.124 | 0.127 | -0.002 (± 0.004) | 0.0% |
| 64 | random | by attitude | 16 | pooled | 0.124 | 0.151 | -0.026 (± 0.013) | 50.0% |
| 64 | random | random | 4 | fixed K/M | 0.124 | 0.124 | +0.001 (± 0.004) | 0.0% |
| 64 | random | random | 4 | pooled | 0.124 | 0.128 | -0.004 (± 0.004) | 12.5% |
| 64 | random | random | 16 | fixed K/M | 0.124 | 0.122 | +0.003 (± 0.004) | 0.0% |
| 64 | random | random | 16 | pooled | 0.124 | 0.142 | -0.017 (± 0.012) | 50.0% |
| 256 | history | by attitude | 4 | fixed K/M | 0.149 | 0.150 | -0.001 (± 0.005) | 0.0% |
| 256 | history | by attitude | 4 | pooled | 0.149 | 0.145 | +0.004 (± 0.005) | 3.1% |
| 256 | history | by attitude | 16 | fixed K/M | 0.149 | 0.149 | -0.000 (± 0.005) | 0.0% |
| 256 | history | by attitude | 16 | pooled | 0.149 | 0.146 | +0.002 (± 0.004) | 12.5% |
| 256 | history | by attitude | 64 | fixed K/M | 0.149 | 0.154 | -0.005 (± 0.005) | 0.0% |
| 256 | history | by attitude | 64 | pooled | 0.149 | 0.146 | +0.003 (± 0.005) | 50.0% |
| 256 | history | random | 4 | fixed K/M | 0.149 | 0.145 | +0.004 (± 0.005) | 0.0% |
| 256 | history | random | 4 | pooled | 0.149 | 0.146 | +0.003 (± 0.005) | 3.1% |
| 256 | history | random | 16 | fixed K/M | 0.149 | 0.141 | +0.008 (± 0.005) | 0.0% |
| 256 | history | random | 16 | pooled | 0.149 | 0.141 | +0.008 (± 0.005) | 12.5% |
| 256 | history | random | 64 | fixed K/M | 0.149 | 0.155 | -0.006 (± 0.005) | 0.0% |
| 256 | history | random | 64 | pooled | 0.149 | 0.145 | +0.004 (± 0.006) | 50.0% |
| 256 | random | by attitude | 4 | fixed K/M | 0.124 | 0.124 | +0.000 (± 0.002) | 0.0% |
| 256 | random | by attitude | 4 | pooled | 0.124 | 0.135 | -0.011 (± 0.003) | 3.1% |
| 256 | random | by attitude | 16 | fixed K/M | 0.124 | 0.125 | -0.001 (± 0.001) | 0.0% |
| 256 | random | by attitude | 16 | pooled | 0.124 | 0.139 | -0.015 (± 0.003) | 12.5% |
| 256 | random | by attitude | 64 | fixed K/M | 0.124 | 0.126 | -0.002 (± 0.002) | 0.0% |
| 256 | random | by attitude | 64 | pooled | 0.124 | 0.137 | -0.013 (± 0.004) | 50.0% |
| 256 | random | random | 4 | fixed K/M | 0.124 | 0.124 | -0.000 (± 0.002) | 0.0% |
| 256 | random | random | 4 | pooled | 0.124 | 0.126 | -0.001 (± 0.002) | 3.1% |
| 256 | random | random | 16 | fixed K/M | 0.124 | 0.125 | -0.001 (± 0.002) | 0.0% |
| 256 | random | random | 16 | pooled | 0.124 | 0.129 | -0.005 (± 0.003) | 12.5% |
| 256 | random | random | 64 | fixed K/M | 0.124 | 0.124 | -0.000 (± 0.002) | 0.0% |
| 256 | random | random | 64 | pooled | 0.124 | 0.138 | -0.014 (± 0.004) | 50.0% |

### Version (b): what deterring one group is worth (every other group complying; always-D vs always-C)

| N | M | group size | to the whole commons | to that group itself | to the other groups |
|---|---|---|---|---|---|
| 64 | 1 | 64 | +2304.0 | +2304.0 | +0.0 |
| 64 | 4 | 16 | +714.4 | -98.3 | +812.6 |
| 64 | 16 | 4 | +21.5 | -138.7 | +160.1 |
| 256 | 1 | 256 | +9216.0 | +9216.0 | +0.0 |
| 256 | 4 | 64 | +2857.4 | -393.0 | +3250.5 |
| 256 | 16 | 16 | +85.8 | -554.6 | +640.4 |
| 256 | 64 | 4 | -7.4 | -155.3 | +147.9 |

N = 64: defection rate 0.457, final stock 8.4 of 256

N = 256: defection rate 0.466, final stock 3.2 of 1024
