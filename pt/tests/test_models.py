import torch

from vmz.models import r2plus1d_34, r2plus1d_152, ip_csn_152, ir_csn_152


def main():
    test_input = torch.rand(4, 3, 36, 112, 112)
    # test non-pretrained models
    model = r2plus1d_34()
    print(model(test_input).size())

    model = r2plus1d_152()
    print(model(test_input).size())

    test_input = torch.rand(4, 3, 36, 224, 224)
    model = ip_csn_152()
    print(model(test_input).size())

    test_input = torch.rand(4, 3, 36, 224, 224)
    model = ir_csn_152()
    print(model(test_input).size())


if __name__ == "__main__":
    main()
