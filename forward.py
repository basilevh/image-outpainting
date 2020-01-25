# Basile Van Hoorick, Jan 2020
'''
Hallucinates beyond all four edges of an image, increasing both dimensions by 50%.
The outpainting process interally converts 128x128 to 192x192, after which the generated output is upscaled.
Then, the original input is blended onto the result for optimal fidelity.
Example usage:
python forward.py input.jpg output.jpg
'''

if __name__ == '__main__':

    import sys
    from outpainting import *

    print("PyTorch version: ", torch.__version__)
    print("Torchvision version: ", torchvision.__version__)

    src_file = sys.argv[1]
    dst_file = sys.argv[2]
    gen_model = load_model('generator_final.pt')
    print('Source file: ' + src_file + '...')
    input_img = plt.imread(src_file)[:, :, :3]
    output_img, blended_img = perform_outpaint(gen_model, input_img)
    plt.imsave(dst_file, blended_img)
    print('Destination file: ' + dst_file + ' written')
