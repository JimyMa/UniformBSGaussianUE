from udntools.region.service_region import ServiceRegion
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


if __name__ == '__main__':
    service_region = ServiceRegion(0.0, 100.0, 0.0, 100.0, 250, 1000)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    for i in service_region.bs_ue_dict:
        for index in service_region.bs_ue_dict[i]:
            plt.scatter(service_region.ue_position_[[0], service_region.bs_ue_dict[i]],
                        service_region.ue_position_[[1], service_region.bs_ue_dict[i]],
                        s=5, marker='o')
    vor = Voronoi(service_region.bs_position_)
    voronoi_plot_2d(vor, show_vertices=False, show_points=False, ax=ax)
    plt.xlim(service_region.x_min, service_region.x_max)
    plt.ylim(service_region.y_min, service_region.y_max)
    plt.show()
