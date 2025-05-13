import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np


class PrecondDataset(Dataset):
    def __init__(self, generator,
                 mode={'img_train', 'img_val', 'qna_train', 'qna_val'},
                 num_samples=1000,
                 transform=None):

        assert mode in {'img_train', 'img_val', 'qna_train', 'qna_val'}

        self.mode = mode

        self.generator = generator
        self.num_samples = num_samples

        self.transform = transform or transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):

        if self.mode == 'img_train' or self.mode == 'qna_train':
            return self.num_samples

        if self.mode == 'img_val' or self.mode == 'qna_val':
            return self.generator.valid_num

    def tensor_convert(self, img, questions, answer):

        # Tensor conversions
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        questions_tensor = torch.tensor(questions, dtype=torch.float32)
        answer_tensor = torch.tensor(answer.argmax(), dtype=torch.long)

        return img_tensor, questions_tensor, answer_tensor

    def __getitem__(self, idx):

        if self.mode == 'img_train':
            img = self.generator.get_image()  # returns np.array
            img = self.transform(img.astype(np.float32))

            return img

        if self.mode == 'img_val':
            img = self.generator.get_image_valid()  # returns np.array
            img = self.transform(img.astype(np.float32))

            return img

        if self.mode == 'qna_train':
            img, questions, answer = self.generator.get_qna()

            return self.tensor_convert(img, questions, answer)

        if self.mode == 'qna_val':
            img, questions, answer = self.generator.get_qna()

            return self.tensor_convert(img, questions, answer)


class PracTestDataset(Dataset):
    def __init__(self, quiz_generator, example_generator=None,
                 mode={'qna_prac', 'qna_test'},
                 transform=None, prob_practice=0.1):

        assert mode in {'qna_prac', 'qna_test'}

        self.mode = mode
        self.generator = quiz_generator

        self.transform = transform or transforms.Compose([
            transforms.ToTensor()
        ])

        if type(example_generator) != type(None):
            self.example_generator = example_generator
            self.spike_precond = True

        else:
            self.spike_precond = False

    def __len__(self):
        if self.mode == 'qna_prac':
            return self.generator.lengths['practice']
        if self.mode == 'qna_test':
            return self.generator.lengths['test']

    def tensor_convert(self, img, questions, answer):

        # Tensor conversions
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        questions_tensor = torch.tensor(np.asarray(questions), dtype=torch.float32)
        answer_tensor = torch.tensor(np.asarray(answer).argmax(), dtype=torch.long)
        return [img_tensor, questions_tensor, answer_tensor]

    def __getitem__(self, idx):

        if self.mode == 'qna_prac':
            img, questions, answer, program, options = self.generator.get_qna(phase='practice')
            return self.tensor_convert(img, questions, answer) + [program, options]

        if self.mode == 'qna_test':
            img, questions, answer, program, options = self.generator.get_qna(phase='test')
            return self.tensor_convert(img, questions, answer) + [program, options]


