from numpy import var
from initial_checks import *
from hsv_color_space import *
from obtain_luminance import *
from skew_kurt_calculation import *
from avg_gradient_magnitude import *
from sobel_operator import *
from hough_operator import *
from lbp_operator import *
from entropy_calculator import *


# Print table
def print_table(title_table, mode_hue, median_sat, median_val,avgLy, varLy, skewness, kurt, asg, sobel, hough, moda, entropy):

    rows = ('HSV: Mode Hue', 'HSV: Median saturation', 'HSV: Median luminance', 'Avg log grayscale', \
        'Variance log grayscale', 'Skewness grayscale', 'Excess kurtosis grayscale',\
        'Average Square Gradients', 'Sobel operator', 'Hough lines',\
        'Mode LBP', 'Shannon entropy')
        
    colors = plt.cm.Oranges(np.linspace(0, 0.5, len(rows)))
    _, axs = plt.subplots(1, 1)
    axs.axis('tight')
    axs.axis('off')
    tab = [[mode_hue], [median_sat], [median_val],[avgLy], [varLy], [skewness], [kurt], [asg], [sobel], [hough], [moda], [entropy]]
    tab_2 = [['%.2f' % j for j in i] for i in tab]
    axs.table(cellText=tab_2,rowLabels = rows, rowColours=colors, colWidths=[0.3, 0.02], loc = 'center')
    axs.set_title(title_table)
    plt.show()  


# One image description -------------------------------------------------------------------------------------------------------------------------------
def params_one_array(src, title_table, show=False, print_all=False):

    (hue_img, sat_img, value_img, mode_hue, median_sat, median_val) = hsv_color(src, show, show)
    (avgLy, varLy) = aux_log_avg_var_luminance(src, show)
    (skewness, kurt) = skewness_kurtosis(src, show)
    asg = aux_calculate_sharpness(src, show)
    sobel = calculate_sobel(src, show)
    hough = hough_operator_func(src, show)
    avg_lbp, median_lbp, min_lbp, max_lbp, moda = calculate_lbp(src, show)
    entropy = entropy_operator(src, show)

    if print_all:
        print('Mode hue =', mode_hue,', Median sat =',median_sat,', Median value =', median_val)
        print('Exp of average of log grayscale =', avgLy, ', Exp of variance of log grayscale:', varLy)
        print('Skewness =', skewness, ', Kurtosis:', kurt)
        print('Average of square gradients = ',asg)
        print('Sobel value = ', sobel)
        print('Hough transform: n° lines = ',hough)
        print('Mode of LBP = ',moda)
        print('Entropy = ', entropy)
    else:
        None

    #print_table(title_table, mode_hue, median_sat, median_val,avgLy, varLy, skewness, kurt, asg, sobel, hough, moda, entropy)

    return (hue_img, sat_img, value_img, mode_hue, median_sat, median_val, \
        avgLy, varLy, \
        skewness, kurt, \
        asg, sobel, hough, moda, entropy)

# All project description (slow) -----------------------------------------------------------------------------------------------------------------------------
def project_description(folder_path, all_images, show=True, x =" ", info = True):
    img_chars = {}
    hue_imgs = []
    sat_imgs = []
    value_imgs = []
    mode_hues = []
    median_sats = []
    median_vals = []
    avgLys = []
    varLys = []
    skewnesses = []
    kurts = []
    asgs = []
    sobels = []
    houghs = []
    modas = []
    entropies = []
    n=0
    v=False
    for f in all_images:
        #print(n)
        
        if f=='Image_000071.jpg':
            v=True if info else False
        else:
            v=False
        
        n = n+1
        #print(f)
        filename = join(folder_path, f)
        src = cv2.imread(filename) #in BGR

        (hue_img, sat_img, value_img, mode_hue, median_sat, median_val, avgLy, varLy, skewness, kurt, asg, sobel, hough, moda, entrop) = params_one_array(src, f, show=v, print_all=v)
        img_chars[f] = (hue_img, sat_img, value_img, mode_hue, median_sat, median_val, avgLy, varLy, skewness, kurt, asg, sobel, hough, moda, entrop)
        hue_imgs.append(hue_img)
        sat_imgs.append(sat_img)
        value_imgs.append(value_img)
        mode_hues.append(mode_hue)
        median_sats.append(median_sat)
        median_vals.append(median_val)
        avgLys.append(avgLy)
        varLys.append(varLy)
        skewnesses.append(skewness)
        kurts.append(kurt)
        asgs.append(asg)
        sobels.append(sobel)
        houghs.append(hough)
        modas.append(moda)
        entropies.append(entrop)



    if info:
        # This is just informative to show images overlapped
        median_hue = np.median(hue_imgs, axis = 0) # Its the median of the hues, you get a 'medianed' image
        median_satu = np.median(sat_imgs, axis = 0)
        median_valu = np.median(value_imgs,  axis = 0)
        
        # This is useful
        fig, ax = plt.subplots(1,4, figsize = (10,5))
        fig.set_figheight(10)
        fig.set_figwidth(25)

        cm = plt.cm.get_cmap('hsv')
        n, bins, patches = ax[0].hist(mode_hues, 30)#, range=[0,255])
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers - min(bin_centers)
        col /= max(col)
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))
        ax[0].set_title("Mode hues histogram")

        cm2 = plt.cm.get_cmap('gist_gray')
        n, bins, patches = ax[1].hist(avgLys, 30)#, range=[0,255])
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers - min(bin_centers)
        col /= max(col)
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm2(c))
        ax[1].set_title("AvgLys histogram")

        ax[2].hist(houghs, bins = 30, color='red')
        ax[2].set_title('Hough lines histogram')

        cm3 = plt.cm.get_cmap('magma')
        n, bins, patches = ax[3].hist(entropies, 30)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = bin_centers - min(bin_centers)
        col /= max(col)
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm3(c))
        ax[3].set_title("Entropies histogram")
        
        varmodeHue = np.var(mode_hues)
        varavgLys = np.var(avgLys)
        varHough = np.var(houghs)
        varEntropy = np.var(entropies)
        #print(varmodeHue, varavgLys, varHough, varEntropy)
        fig.savefig(x)
    else:
        None
    
    return (varmodeHue, varavgLys, varHough, varEntropy, img_chars)
    

"""
folder_path = "/home/asoria/Documents/913440_not_localized/ID913440_images/"
project_description(folder_path)
"""










