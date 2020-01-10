# -*- encoding:UTF-8 -*-
from __future__ import print_function

__author__ = 'jialiang'

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import getopt
import xlsxwriter
import logging


class qdl_analyzer:
    input_folder_path = None
    output_folder_path = None
    camera_flag = True
    sensor_flag = True
    report_flag = True
    plot_flag = True
    accel_interval = 100000
    gyro_interval = 100000
    mag_interval = 5000
    headTracking_interval = 100000
    camera_interval = 3000
    logger = None
    contain_index = True

    def find_qvrdatalogger_path(self, search_camera_folder=True, search_sensors_folder=True):
        camera_folders = []
        sensors_folders = []
        for root, dirs, files in os.walk(self.input_folder_path):
            for name in dirs:
                if search_camera_folder and (name == "Camera" or name == "RGBCamera"):
                    camera_folders.append(os.path.join(root, name))
                elif search_sensors_folder and name == "Sensors":
                    sensors_folders.append(os.path.join(root, name))
        return camera_folders, sensors_folders

    @staticmethod
    def create_folder(dir_path, folder_name):
        folder_path = os.path.join(dir_path, folder_name)
        folder_exists = os.path.exists(folder_path)
        if not folder_exists:
            os.makedirs(folder_path)
            print("Create folder {}".format(folder_path))
        else:
            print("The folder already been created.")

    @staticmethod
    def quar2angle(quar):
        quar = np.array(quar)
        qw, qx, qy, qz = quar[0], quar[1], quar[2], quar[3]
        x_rota = np.rad2deg(np.arctan2(2 * (qw * qx + qy * qz), [1] * len(qx) - 2 * (qx ** 2 + qy ** 2)))
        y_rota = np.rad2deg(np.arcsin(2 * (qw * qy - qz * qx)))
        z_rota = np.rad2deg(np.arctan2(2 * (qw * qz + qx * qy), [1] * len(qx) - 2 * (qy ** 2 + qz ** 2)))
        qdl_analyzer.check_angle_jump(x_rota)
        qdl_analyzer.check_angle_jump(y_rota)
        qdl_analyzer.check_angle_jump(z_rota)
        return x_rota, y_rota, z_rota

    @staticmethod
    def check_angle_jump(angles):
        convert_flag = False
        sign = 1
        THRESHOLD = 300
        different = np.array(pd.DataFrame(angles) - pd.DataFrame(angles).shift(1))
        different = different.T[0]
        for i in range(1, len(different)):
            if different[i] > THRESHOLD or different[i] < -THRESHOLD:
                if convert_flag is False:
                    convert_flag = True
                    if different[i] < -THRESHOLD:
                        sign = 1
                    else:
                        sign = -1
                else:
                    convert_flag = False
            if convert_flag:
                angles[i] = angles[i] + 360*sign

    def check_index_jump(self, index, fname):
        index = np.array(index)
        if index.shape[0] > 1:
            print("Now check index jump")
            self.logger.info("Now check index jump")
            index = np.array(index)
            result = np.array(index[1:] - index[:-1])
            max_value = max(index)
            min_value = min(index)
            result[result <= min_value] += max_value - min_value + 1
            index_jump_counter = 0
            index_miss_counter = 0
            for i in range(result.shape[0]):
                if result[i] != 1:
                    self.logger.critical("Find data sample index jump {} at {} in {}".format(result[i], i+4, fname))
                    print("Find data sample index jump {} at {} in {}".format(result[i], i+4, fname))
                    index_jump_counter += 1
                    index_miss_counter += result[i]
            self.logger.critical("Totally found {} times index jump with {} index missing in file {}.".format(index_jump_counter, index_miss_counter, fname))
            print("Totally found {} times index jump with {} index missing in file {}.".format(index_jump_counter, index_miss_counter, fname))
        return np.array(result), index_jump_counter, index_miss_counter

    def get_frequency(self, timestamp, fname, index):
        index_change = np.ones(len(timestamp) - 1)
        index_jump_counter = 0
        index_miss_counter = 0
        if self.contain_index:
            index_change, index_jump_counter, index_miss_counter = self.check_index_jump(index, fname)
        if len(timestamp) > 1:
            timestamp = np.array(timestamp)
            timestamp_change = np.array(timestamp[1:] - timestamp[:-1])
            frequency = 10**9 * np.array(index_change / timestamp_change)
            threshold = 0.7 * np.average(frequency)
            frequency_drop_counter = 0
            for i in range(frequency.shape[0]):
                if frequency[i] < threshold:
                    print("Frequency drop at line {} to line {} with index change {} and timestamp change {} in file {}".format(i+4, i+5, index_change[i], timestamp_change[i], fname))
                    self.logger.critical("Frequency drop at line {} to line {} with index change {} and timestamp change {} in file {}".format(i+4, i+5, index_change[i], timestamp_change[i], fname))
                    frequency_drop_counter += 1
            self.logger.critical("Totally found {} times frequency drop in file {}.".format(frequency_drop_counter, fname))
            print("Totally found {} times frequency drop in file {}.".format(frequency_drop_counter, fname))
            return frequency, frequency_drop_counter, index_jump_counter, index_miss_counter
        else:
            self.logger.critical("Please check the file {}. The length of timestamp is smaller than 1.".format(fname))
            return None

    @staticmethod
    def data_anaylzer(array):
        percentage = 10
        numer_of_sample = array.shape[1]
        partial_index = int(numer_of_sample / percentage)
        max_overall = np.max(array, axis=1)
        min_overall = np.min(array, axis=1)
        aver_overall = np.average(array, axis=1)
        std_overall = np.std(array, axis=1)
        range_overall = max_overall - min_overall
        num_sample_overall = np.array([int(numer_of_sample)] * array.shape[0])
        overall = np.vstack((aver_overall, std_overall, max_overall, min_overall, range_overall, num_sample_overall))

        max_first = np.max(array[:, :partial_index], axis=1)
        min_first = np.min(array[:, :partial_index], axis=1)
        aver_first = np.average(array[:, :partial_index], axis=1)
        std_first = np.std(array[:, :partial_index], axis=1)
        range_first = max_first - min_first
        num_sample_first = np.array([partial_index] * array.shape[0])
        first_part = np.vstack((aver_first, std_first, max_first, min_first, range_first, num_sample_first))

        max_last = np.max(array[:, -partial_index:], axis=1)
        min_last = np.min(array[:, -partial_index:], axis=1)
        aver_last = np.average(array[:, -partial_index:], axis=1)
        std_last = np.std(array[:, -partial_index:], axis=1)
        range_last = max_last - min_last
        num_sample_last = np.array([partial_index] * array.shape[0])
        last_part = np.vstack((aver_last, std_last, max_last, min_last, range_last, num_sample_last))

        result_data = np.vstack((overall, first_part, last_part))
        return result_data

    @staticmethod
    def frequency_processing(freq):
        frequency = np.insert(freq, 0, freq[0])
        return frequency

    def file_parser(self, file_path, fname, detail_log=True):
        with open(os.path.join(file_path, fname), "r") as fh:
            timestamp = []
            frequency = []
            x_value = []
            y_value = []
            z_value = []
            index = []
            frequency_drop_counter = 0
            index_jump_counter = 0
            index_miss_counter = 0
            timestamp_regression_issue_counter = 0
            if fname != "MetaInfo.xml" and self.sensor_flag:
                if fname == "accelerometer.xml" or fname == "gyroscope.xml" or fname == "magnetometer.xml":
                    x_index = 1
                    timestamp_index = 4
                    taglen = 5
                    z_spliter = "'"
                elif fname == "headTrackingPose.xml":
                    x_index = 6
                    timestamp_index = 5
                    taglen = 9
                    z_spliter = "/"
                    qx = []
                    qy = []
                    qz = []
                    qw = []
                for line in fh:
                    rline = line.split(r"=")
                    if (rline[0].split(" ")[0] == "<Data") and (len(rline) >= taglen) and (
                        rline[-1].split("'")[-1].strip() == "/>"):
                        #print(rline)
                        if x_index != 6:
                            if fname == "magnetometer.xml":
                                x_value.append(float(rline[x_index].split("'")[1])*65536)
                                y_value.append(float(rline[x_index + 1].split("'")[1])*65536)
                                z_value.append(float(rline[x_index + 2].split(z_spliter)[1])*65536)
                            else:
                                x_value.append(float(rline[x_index].split("'")[1]))
                                y_value.append(float(rline[x_index + 1].split("'")[1]))
                                z_value.append(float(rline[x_index + 2].split(z_spliter)[1]))
                        else:
                            x_string = rline[x_index].split(" ")[0]
                            x_value.append(float(x_string.split("'")[1]))
                            y_string = rline[x_index + 1].split(" ")[0]
                            y_value.append(float(y_string.split("'")[1]))
                            z_string = rline[x_index + 2].split(" ")[0]
                            z_value.append(float(z_string.split("'")[1]))
                            qx.append(float(rline[1].split("'")[1]))
                            qy.append(float(rline[2].split("'")[1]))
                            qz.append(float(rline[3].split("'")[1]))
                            qw.append(float(rline[4].split("'")[1]))
                        timestamp.append(float(rline[timestamp_index].split("'")[1]))
                        if self.contain_index:
                            index.append(float(rline[-1].split("'")[1]))

                if len(timestamp) > 1:
                    if len(index) < 1:
                        self.contain_index = False
                    frequency, frequency_drop_counter, index_jump_counter, index_miss_counter = self.get_frequency(timestamp, fname, index)
                if np.array(frequency).shape[0] != 0:
                    for i in range(frequency.shape[0]):
                        if frequency[i] < 0:
                            if detail_log:
                                print("Timestamp regression issue appears at the {}th line of the {} file.".format(i + 4, fname))
                                self.logger.critical("Timestamp regression issue appears at the {}th line of the {} file.".format(i + 4, fname))
                            timestamp_regression_issue_counter += 1
                    if detail_log:
                        print("Totally found {} times of timestamp regression issue in the {} file.".format(timestamp_regression_issue_counter, fname))
                        self.logger.critical("Totally found {} times of timestamp regression issue in the {} file.".format(timestamp_regression_issue_counter, fname))

                for i in range(len(x_value)-1):
                    if x_value[i] == 0:
                        if y_value[i] == 0:
                            if z_value[i] == 0:
                                if detail_log:
                                    print("WARNING: Find zero value on x, y, z in line {} of file {}.".format(i + 4, fname))
                                    self.logger.warning("Find zero value on x, y, z in line {} of file {}.".format(i + 4, fname))
                            else:
                                if detail_log:
                                    print("WARNING: Find zero value on x, y in line {} of file {}.".format(i + 4, fname))
                                    self.logger.warning("Find zero value on x, y in line {} of file {}.".format(i + 4, fname))
                        else:
                            if z_value[i] == 0:
                                if detail_log:
                                    print("WARNING: Find zero value on x, z in line {} of file {}.".format(i + 4, fname))
                                    self.logger.warning("Find zero value on x, z in line {} of file {}.".format(i + 4, fname))
                            else:
                                pass
                    else:
                        if y_value[i] == 0:
                            if z_value[i] == 0:
                                if detail_log:
                                    print("WARNING: Find zero value on y, z in line {} of file {}.".format(i + 4, fname))
                                    self.logger.warning("Find zero value on y, z in line {} of file {}.".format(i + 4, fname))
                            else:
                                pass
                        else:
                            if z_value[i] == 0:
                                pass
                issue_summary = list([timestamp_regression_issue_counter, frequency_drop_counter, index_jump_counter, index_miss_counter])
                if fname == "headTrackingPose.xml":
                    quar = np.array((qw, qx, qy, qz))
                    x_rota, y_rota, z_rota = self.quar2angle(quar)
                    return timestamp, frequency, x_value, y_value, z_value, x_rota, y_rota, z_rota, issue_summary
                else:
                    return timestamp, frequency, x_value, y_value, z_value, issue_summary
            if fname == "MetaInfo.xml" and self.camera_flag:
                for line in fh:
                    rline = line.split(r"=")
                    if (rline[0].split(" ")[0] == "<Frame") and (len(rline) >= 3) and (
                        rline[-1].split(r"/")[-1] == ">\n"):
                        timestamp.append(float(rline[2].split("'")[1]))
                df = pd.DataFrame({'timestamp': timestamp})
                frequency = np.array(1 / ((df['timestamp'] - df['timestamp'].shift(1)) / 1000000) * 1000)
                timestamp_regression_issue_counter = 0
                for i in range(frequency.shape[0]):
                    if frequency[i] < 0:
                        if detail_log:
                            print("Timestamp regression issue appears at the {}th line of the {} file.".format(i + 4, fname))
                            self.logger.critical("Timestamp regression issue appears at the {}th line of the {} file.".format(i + 4, fname))
                        timestamp_regression_issue_counter += 1
                if detail_log:
                    print("Totally found {} timestamp regression issue in the {} file.".format(timestamp_regression_issue_counter, fname))
                    self.logger.critical("Totally found {} timestamp regression issue in the {} file.".format(timestamp_regression_issue_counter, fname))
                return timestamp, frequency

    @staticmethod
    def plot_time_info(save_path, title, data, type="t"):
        if type == "f":
            plt.xlabel("Num of Samples")
            plt.ylabel("Frequency")
            plt.title(title)
            plt.plot(data, label="Frequency-to-Samples")
        else:
            plt.xlabel("Timestamp")
            plt.ylabel("Number of Samples")
            plt.title(title)
            plt.plot(data, range(len(data)), label="Timestamp-to-Samples")
        plt.legend(loc="best")
        plt.draw()
        plt.savefig(os.path.join(save_path, title + ".png"))
        plt.clf()
        return 0

    @staticmethod
    def plot_sample_value(save_path, title, x, y, z, ts):
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("value")
        plt.plot(ts, x, label="x-axis")
        plt.plot(ts, y, label="y-axis")
        plt.plot(ts, z, label="z-axis")
        plt.legend(loc="best")
        plt.draw()
        plt.savefig(os.path.join(save_path, title + ".png"))
        plt.clf()
        return 0

    def ploting_process(self, file_path, fname):
        output_save_folder = os.path.join(self.output_folder_path, "analysis_log_" + file_path.split("\\")[-2])
        self.create_folder(self.output_folder_path, "analysis_log_" + file_path.split("\\")[-2])
        if fname == "accelerometer.xml" and self.sensor_flag:
            interval = self.accel_interval
            timestamp, frequency, x_value, y_value, z_value, issue_summary = self.file_parser(file_path, fname, detail_log=False)
            name = fname.split(".")[0]
            self.create_folder(output_save_folder, "accelerometer_data_plots")
            save_path = os.path.join(output_save_folder, "accelerometer_data_plots")
            print("Plotting {} value overall-figure.".format(name))
            self.logger.info("Plotting {} value overall-figure.".format(name))
            self.plot_sample_value(save_path, "{}_value(overall)".format(name), x_value, y_value, z_value, timestamp)
        elif fname == "gyroscope.xml" and self.sensor_flag:
            interval = self.gyro_interval
            timestamp, frequency, x_value, y_value, z_value, issue_summary = self.file_parser(file_path, fname, detail_log=False)
            name = fname.split(".")[0]
            self.create_folder(output_save_folder, "gyroscope_data_plots")
            save_path = os.path.join(output_save_folder, "gyroscope_data_plots")
            print("Plotting {} value overall-figure.".format(name))
            self.logger.info("Plotting {} value overall-figure.".format(name))
            self.plot_sample_value(save_path, "{}_value(overall)".format(name), x_value, y_value, z_value, timestamp)
        elif fname == "magnetometer.xml" and self.sensor_flag:
            interval = self.mag_interval
            timestamp, frequency, x_value, y_value, z_value, issue_summary = self.file_parser(file_path, fname, detail_log=False)
            name = fname.split(".")[0]
            self.create_folder(output_save_folder, "magnetometer_data_plots")
            save_path = os.path.join(output_save_folder, "magnetometer_data_plots")
            print("Plotting {} value overall-figure.".format(name))
            self.logger.info("Plotting {} value overall-figure.".format(name))
            self.plot_sample_value(save_path, "{}_value(overall)".format(name), x_value, y_value, z_value, timestamp)
        elif fname == "headTrackingPose.xml" and self.sensor_flag:
            interval = self.headTracking_interval
            timestamp, frequency, x_value, y_value, z_value, x_rota, y_rota, z_rota, issue_summary = self.file_parser(file_path, fname, detail_log=False)
            name = fname.split(".")[0]
            self.create_folder(output_save_folder, "headTrackingPose_plots")
            save_path = os.path.join(output_save_folder, "headTrackingPose_plots")
            print("Plotting {} position overall-figure.".format(name))
            self.logger.info("Plotting {} position overall-figure.".format(name))
            self.plot_sample_value(save_path, "{}_position_value(overall)".format(name), x_value, y_value, z_value,
                                   timestamp)
            print("Plotting {} orientation overall-figure.".format(name))
            self.logger.info("Plotting {} orientation overall-figure.".format(name))
            self.plot_sample_value(save_path, "{}_orientation_value(overall)".format(name), x_rota, y_rota, z_rota,
                                   timestamp)
        elif fname == "MetaInfo.xml" and self.camera_flag:
            interval = self.camera_interval
            timestamp, frequency = self.file_parser(file_path, fname, detail_log=False)
            self.create_folder(output_save_folder, "Camera_data_plots")
            save_path = os.path.join(output_save_folder, "Camera_data_plots")
            name = "Camera_frame"
        else:
            print("Error: no corresponding file name: {} was found.".format(fname))
            self.logger.error("Error: no corresponding file name: {} was found.".format(fname))
            return 0
        print("{} average frequency is {} hz.".format(name, sum(frequency[1:]) / (len(frequency) - 1)))
        self.logger.info("{} average frequency is {} hz.".format(name, sum(frequency[1:]) / (len(frequency) - 1)))
        num_subplot = int(math.ceil(len(timestamp) / interval))
        for m in range(int(len(timestamp) / interval)):
            end_index = interval * (m + 1)
            if interval * (m + 1) > len(timestamp):
                end_index = len(timestamp)
            if m == 0:
                print("Plotting {} interval-figures of {} frequency.".format(num_subplot, name))
                self.logger.info("Plotting {} interval-figures of {} frequency.".format(num_subplot, name))
            self.plot_time_info(save_path, "{}_frequency_{}".format(name, m), frequency[interval * m:end_index],
                                type="f")
            if m == 0:
                print("Plotting {} interval-figures of {} timestamp.".format(num_subplot, name))
                self.logger.info("Plotting {} interval-figures of {} timestamp.".format(num_subplot, name))
            self.plot_time_info(save_path, "{}_timestamp_{}".format(name, m), timestamp[interval * m:end_index])

            if fname != "MetaInfo.xml" and self.sensor_flag:
                subplot_mid_name = ""
                if fname == "headTrackingPose.xml" and self.sensor_flag:
                    if m == 0:
                        print("Plotting {} interval-figures of {} orientation value.".format(num_subplot, name))
                        self.logger.info("Plotting {} interval-figures of {} orientation value.".format(num_subplot, name))
                    self.plot_sample_value(save_path,
                                           "{}_orientation_value_{}".format(name, m),
                                           x_rota[interval * m:end_index],
                                           y_rota[interval * m:end_index],
                                           z_rota[interval * m:end_index],
                                           timestamp[interval * m:end_index])
                    subplot_mid_name = "position"
                if m == 0:
                    print("Plotting {} interval-figures of {} {} value.".format(num_subplot, name, subplot_mid_name))
                    self.logger.info("Plotting {} interval-figures of {} {} value.".format(num_subplot, name, subplot_mid_name))
                self.plot_sample_value(save_path,
                                       "{}_{}_value_{}".format(name, subplot_mid_name, m),
                                       x_value[interval * m:end_index],
                                       y_value[interval * m:end_index],
                                       z_value[interval * m:end_index],
                                       timestamp[interval * m:end_index])
        print("Plotting {}_frequency(overall) figure.".format(name))
        self.logger.info("Plotting {}_frequency(overall) figure.".format(name))
        self.plot_time_info(save_path, "{}_frequency(overall)".format(name), frequency, type="f")
        print("Plotting {}_timestamp(overall) figure.".format(name))
        self.logger.info("Plotting {}_timestamp(overall) figure.".format(name))
        self.plot_time_info(save_path, "{}_timestamp(overall)".format(name), timestamp)

    @staticmethod
    def create_data_summary_xlsx(folder_path, data_pack, data_name, percentage=10):
        # Create a workbook and add a worksheet.
        workbook = xlsxwriter.Workbook(os.path.join(folder_path, "{}_summary.xlsx".format(data_name)))
        worksheet = workbook.add_worksheet()
        title_format = workbook.add_format({'align': 'center', 'bold': True, 'valign': 'vcenter'})
        title_format.set_border()
        title_format.set_bg_color("#CCCCCC")
        data_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
        data_format.set_border()
        link_format = workbook.add_format(
            {'align': 'center', 'valign': 'vcenter', 'underline': 1, 'font_color': 'blue'})
        link_format.set_border()
        worksheet.set_column('A:B', 18)
        worksheet.merge_range('A2:B2', "  ", title_format)
        worksheet.merge_range('A3:B3', "  ", title_format)
        worksheet.merge_range('A4:A9', "Overall", title_format)
        worksheet.merge_range('A10:A15', r"First {}% Samples".format(percentage), title_format)
        worksheet.merge_range('A16:A21', r"Last {}% Samples".format(percentage), title_format)
        row_index = 3
        col_index = 1
        for i in range(3):
            worksheet.write_string(row_index + i * 6, col_index, "Average", cell_format=title_format)
            worksheet.write_string(row_index + i * 6 + 1, col_index, "Std_dev", cell_format=title_format)
            worksheet.write_string(row_index + i * 6 + 2, col_index, "Max", cell_format=title_format)
            worksheet.write_string(row_index + i * 6 + 3, col_index, "Min", cell_format=title_format)
            worksheet.write_string(row_index + i * 6 + 4, col_index, "Range", cell_format=title_format)
            worksheet.write_string(row_index + i * 6 + 5, col_index, "Number of Sample", cell_format=title_format)
        col_starting_index = 2
        for (name, data) in data_pack:
            row_starting_index = 2
            if data.shape[1] > 1:
                worksheet.merge_range(row_starting_index - 1,
                                      col_starting_index,
                                      row_starting_index - 1,
                                      col_starting_index + data.shape[1] - 1,
                                      name.split(".")[0],
                                      title_format)
            else:
                worksheet.write_string(row_starting_index - 1,
                                       col_starting_index,
                                       name.split(".")[0],
                                       title_format)
            worksheet.set_column(col_starting_index, col_starting_index + data.shape[1] - 1, 18)
            if data.shape[1] == 4:
                title_list = ["Frequency(Hz)", "X_value", "Y_value", "Z_value"]
            elif data.shape[1] == 7:
                title_list = ["Frequency(Hz)", "Position_X(mm)", "Position_Y(mm)", "Position_Z(mm)",
                              u"Orientation_X(°)", u"Orientation_Y(°)", u"Orientation_Z(°)"]
            else:
                title_list = ["Frequency(Hz)"]
            worksheet.write_row(row_starting_index, col_starting_index, title_list, cell_format=title_format)
            for i in range(data.shape[0]):
                worksheet.write_row(row_starting_index + 1 + i, col_starting_index, data[i], cell_format=data_format)
            col_starting_index += data.shape[1]
        workbook.close()

    @staticmethod
    def create_issue_summary_xlsx(folder_path, data_pack, data_name):
        workbook = xlsxwriter.Workbook(os.path.join(folder_path, "{}_issue_detection_summary.xlsx".format(data_name)))
        worksheet = workbook.add_worksheet()
        title_format = workbook.add_format({'align': 'center', 'bold': True, 'valign': 'vcenter'})
        title_format.set_border()
        title_format.set_bg_color("#CCCCCC")
        worksheet.set_column('A:E', 20)
        data_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
        data_format.set_border()
        link_format = workbook.add_format(
            {'align': 'center', 'valign': 'vcenter', 'underline': 1, 'font_color': 'blue'})
        link_format.set_border()
        row_index = 0
        col_index = 1
        worksheet.write_string(row_index, col_index, "Timestamp Regression", cell_format=title_format)
        worksheet.write_string(row_index, col_index + 1, "Frequency Drop", cell_format=title_format)
        worksheet.write_string(row_index, col_index + 2, "Index Jump", cell_format=title_format)
        worksheet.write_string(row_index, col_index + 3, "Index Miss", cell_format=title_format)
        for (name, data) in data_pack:
            worksheet.write_string(row_index + 1, col_index - 1, name, cell_format=title_format)
            worksheet.write_row(row_index + 1, col_index, data, cell_format=data_format)
            row_index += 1
        workbook.close()

    def create_all_plots(self):
        camera_folders, sensors_folders = self.find_qvrdatalogger_path()
        current_dir = os.getcwd()
        if self.sensor_flag:
            for sensor_case in sensors_folders:
                print("========================================")
                self.logger.info("========================================")
                print("Now: {}".format(sensor_case))
                self.logger.info("Now: {}".format(sensor_case))
                os.chdir(sensor_case)
                files = glob.glob("*.xml")
                for fname in files:
                    self.ploting_process(sensor_case, fname)
                os.chdir(current_dir)
        if self.camera_flag:
            for camera_case in camera_folders:
                print("========================================")
                self.logger.info("========================================")
                print("Now: {}".format(camera_case))
                self.logger.info("Now: {}".format(camera_case))
                os.chdir(camera_case)
                files = glob.glob("*.xml")
                for fname in files:
                    self.ploting_process(camera_case, fname)
                os.chdir(current_dir)

    def build_report(self):
        camera_folders, sensors_folders = self.find_qvrdatalogger_path()
        current_dir = os.getcwd()
        if self.sensor_flag:
            for sensor_case in sensors_folders:
                print("========================================")
                self.logger.info("========================================")
                print("Now creating report for: {}".format(sensor_case))
                self.logger.info("Now creating report for: {}".format(sensor_case))
                data_pack = []
                issue_data_pack = []
                os.chdir(sensor_case)
                files = glob.glob("*.xml")
                for file_name in files:
                    if file_name == "headTrackingPose.xml":
                        timestamp, freq, x_value, y_value, z_value, x_rota, y_rota, z_rota, issue_summary = \
                            self.file_parser(sensor_case, file_name)
                        if np.array(freq).shape[0] != 0:
                            frequency = self.frequency_processing(freq)
                            result_data = self.data_anaylzer(
                                np.array((frequency, x_value, y_value, z_value, x_rota, y_rota, z_rota)))
                            data_pack.append((file_name, result_data))
                            issue_data_pack.append((file_name, issue_summary))
                        else:
                            print("File {} is empty. Skip it.".format(file_name))
                            self.logger.info("File {} is empty. Skip it.".format(file_name))
                            continue
                    elif file_name == "gyroscope.xml" or "accelerometer.xml" or "magnetometer.xml":
                        timestamp, freq, x_value, y_value, z_value, issue_summary = \
                            self.file_parser(sensor_case, file_name)
                        if np.array(freq).shape[0] != 0:
                            frequency = self.frequency_processing(freq)
                            result_data = self.data_anaylzer(np.array((frequency, x_value, y_value, z_value)))
                            data_pack.append((file_name, result_data))
                            issue_data_pack.append((file_name, issue_summary))
                        else:
                            print("File {} is empty. Skip it.".format(file_name))
                            self.logger.info("File {} is empty. Skip it.".format(file_name))
                            continue
                output_save_folder = os.path.join(self.output_folder_path, "analysis_log_" + sensor_case.split("\\")[-2])
                self.create_folder(self.output_folder_path, "analysis_log_" + sensor_case.split("\\")[-2])
                self.create_data_summary_xlsx(output_save_folder, data_pack, sensor_case.split("\\")[-1])
                self.create_issue_summary_xlsx(output_save_folder, issue_data_pack, sensor_case.split("\\")[-1])
                os.chdir(current_dir)
        if self.camera_flag:
            for camera_case in camera_folders:
                print("========================================")
                self.logger.info("========================================")
                print("Now creating report for: {}".format(camera_case))
                self.logger.info("Now creating report for: {}".format(camera_case))
                os.chdir(camera_case)
                files = glob.glob("*.xml")
                data_pack = []
                for file_name in files:
                    timestamp, freq = self.file_parser(camera_case, file_name)
                    frequency = np.array(freq)
                    frequency[0] = frequency[1]
                    frequency = frequency.reshape((1, frequency.shape[0]))
                    result_data = self.data_anaylzer(frequency)
                    data_pack.append(("Camera_frame", result_data))
                output_save_folder = os.path.join(self.output_folder_path, "analysis_log_"+camera_case.split("\\")[-2])
                self.create_folder(self.output_folder_path, "analysis_log_"+camera_case.split("\\")[-2])
                self.create_data_summary_xlsx(output_save_folder, data_pack, camera_case.split("\\")[-1])
                os.chdir(current_dir)

    def input_argument(self):
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:c:s:a:d:g:p:r:n:cf:")
        for op, value in opts:
            if op == "-i":
                self.input_folder_path = value
                if self.input_folder_path is None:
                    print("Error: input folder is missing. -h : for more information.")
                    sys.exit()
            elif op == "-o":
                self.output_folder_path = value
                if self.output_folder_path is None:
                    self.output_folder_path = self.input_folder_path
                    print("Using the input folder as default output folder.")
            elif op == "-c":
                if value == "1":
                    self.camera_flag = True
                elif value == "0":
                    self.camera_flag = False
                else:
                    print("Error: wrong type of -c value. -h : for more information.")
                    sys.exit()
            elif op == "-s":
                if value == "1":
                    self.sensor_flag = True
                elif value == "0":
                    self.sensor_flag = False
                else:
                    print("Error: wrong type of -s value. -h : for more information.")
                    sys.exit()
            elif op == "-d":
                if value == "1":
                    self.plot_flag = True
                elif value == "0":
                    self.plot_flag = False
                else:
                    print("Error: wrong type of -d value. -h : for more information.")
                    sys.exit()
            elif op == "-r":
                if value == "1":
                    self.report_flag = True
                elif value == "0":
                    self.report_flag = False
                else:
                    print("Error: wrong type of -r value. -h : for more information.")
                    sys.exit()
            elif op == "-a":
                self.accel_interval = int(value)
            elif op == "-g":
                self.gyro_interval = int(value)
            elif op == "-m":
                self.mag_interval = int(value)
            elif op == "-p":
                self.headTracking_interval = int(value)
            elif op == "-f":
                self.camera_interval = int(value)
            elif op == "-n":
                if value == "1":
                    self.contain_index = True
                elif value == "0":
                    self.contain_index = False
                else:
                    print("Error: wrong type of -n value. -h : for more information.")
                    sys.exit()
            elif op == "-h":
                print("================================================================")
                print(" -i : input folder.")
                print(" -o : output folder. Default value is set to the input folder.")
                print(" -r : set report_flag. whether output data report. Default value is 1(Output), set 0 to close.")
                print(" -d : set plot_flag. whether draw data figures. Default value is 1(Draw), set 0 to close.")
                print(" -a : set the number of accelerometer samples on each interval plots. Default value: {}".format(10000))
                print(" -g : set the number of gyro samples on each interval plots. Default value: {}".format(10000))
                print(" -m : set the number of magnetometer samples on each interval plots. Default value: {}".format(5000))
                print(" -p : set the number of headTracking samples on each interval plots. Default value: {}".format(
                    10000))
                print(" -f : set the number of camera frames on each interval plots. Default value: {}".format(3000))
                print(" -c : set camera_flag. Whether you want to start draw figures regarding the camera data.")
                print("      0: to close. Default: open")
                print(" -s : set sensor_flag. Whether you want to start draw figures regarding the sensor data.")
                print("      0: to close. Default: open")
                print(" -n : set contain_index. Whether the qvrdatalogger log files contains count of sample index. The default value is True. Set 1 for True， 0 for false.")
                print(" -h : for help.")
                print("================================================================")
                sys.exit()

            if self.input_folder_path != None:
                if self.output_folder_path == None:
                    self.output_folder_path = self.input_folder_path
                if not os.path.exists(self.output_folder_path):
                    os.makedirs(self.output_folder_path)
                logging.basicConfig(filename=os.path.join(self.output_folder_path, "analyzer.log"), filemode="w",
                                    format="%(asctime)s:%(levelname)s:%(message)s",
                                    datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
                self.logger = logging.getLogger(__name__)

    def run_analyzer(self):
        self.input_argument()
        if self.report_flag:
            print("===========START CREATING REPORT============")
            self.logger.info("===========START CREATING REPORT============")
            self.build_report()
        if self.plot_flag:
            print("===========START CREATING FIGURES============")
            self.logger.info("===========START CREATING FIGURES============")
            self.create_all_plots()


if __name__ == "__main__":
    test = qdl_analyzer()
    test.run_analyzer()
