import argparse, os
import numpy as np
import torch
import U_net_Model as Model
import SimpleITK as sitk

def load_model():
    input_channels = 5
    out_channels = 2
    model = Model.UNet(in_channels=input_channels, out_channels=out_channels)
    model_dict = torch.load('./model_lung/best_model.pth')["state_dict"]
    
    model.load_state_dict(model_dict)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    return model



def get_model_input(ct_array,resolution):
    wc,ww = -600,1600
    ct_array = (ct_array - wc) / ww
    sample_list = []
    z,y,x = ct_array.shape
    window = (-5, -2, 0, 2, 5)
    for slice_index in range(z):
        sample = np.zeros([len(window), y, x], 'float32')
        for idx in range(len(window)):
            slice_id = slice_index + int(window[idx]/resolution[0])
            if slice_id >= 0 and slice_id < z:
                sample[idx,:,:] = ct_array[slice_id,:,:]
        sample_list.append(sample)
    return sample_list



def predict(test_model, sample_list):
    sample_array = np.stack(sample_list, axis=0) # z * 5 * y * x
    batch_size = 8
    prediction_list = []
    index = 0
    soft_max = torch.nn.Softmax(dim=1)
    test_model.eval()
    with torch.no_grad():
        while index < len(sample_list):
            index_end = index + batch_size
            if index_end >= len(sample_list):
                index_end = len(sample_list)
            inputs = torch.from_numpy(sample_array[index: index_end, :, :, :]).cuda()
            prediction = test_model(inputs)
            prediction = soft_max(prediction)
            prediction = prediction.cpu().numpy() # batch_size * 2 * y * x
            prediction_list.append(prediction)
            index = index_end
    prediction_array = np.concatenate(prediction_list, axis=0) # z * 2 * y * x
    lung_mask = np.array(prediction_array[:,1,:,:] > 0.5,'float32')
    return lung_mask



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='lung segmentation of a ct volume')
    parser.add_argument('--input_dir', default='', type=str, metavar='PATH',
                            help='this directory contains all test samples(ct volumes)')
    parser.add_argument('--predict_dir', default='', type=str, metavar='PATH',
                            help='segmentation file of each test sample should be stored in the directory')

    args = parser.parse_args()
    input_dir =  args.input_dir
    predict_dir = args.predict_dir

    test_model = load_model()
    print("model loaded successfully!")

    for ct_file in os.listdir(input_dir):
        input_file = os.path.join(input_dir,ct_file)
        dataname = ct_file.split('\.')[0]

        input_image = sitk.ReadImage(input_file)
        input_array = sitk.GetArrayFromImage(input_image)
        resolution = input_image.GetSpacing()
        resolution = (resolution[2],resolution[1],resolution[0])

        sample_list = get_model_input(input_array,resolution)
        print("start predicting input volume!",dataname)
        lung_mask = predict(test_model,sample_list)

        mask_image = sitk.GetImageFromArray(lung_mask)
        mask_image.SetOrigin(input_image.GetOrigin())
        mask_image.SetSpacing(input_image.GetSpacing())

        sitk.WriteImage(mask_image,os.path.join(predict_dir,dataname+'.nii.gz'))
        print("segmentation is generated successfully!",dataname)






    



