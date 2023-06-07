import torch
from torchvision import transforms

labels = ['cataract', 'diabetic retinopathy', 'glaucoma', 'normal']

def predict(model, image):
    try :
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.42045337, 0.27952126, 0.1714828],
                [0.28596467, 0.20711626, 0.15339227]
            )
        ])

        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

            all_prob, all_id = torch.topk(probabilities, 2)

            predictions = []
            for i in range(all_prob.size(0)):
                prob = all_prob[i].item()
                label = all_id[i].item()
                predictions.append((prob, label))

            return predictions
    except Exception as e:
        print(f"Error predicting image: {e}")
        return []
