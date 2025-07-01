#pragma once
#include <fstream>
#include <iomanip>
#include <iostream>

template <class T>
int PrintData2Txt(T *data, const char *path, int height, int width,
                  int stride = 0) {
  using namespace std;
  stride = stride < width ? width : stride;
  int ret = -1;
  // using type = constexpr(std::is_same_v<T, half>) ? float : T;
  using type = float;
  ofstream os(path);
  if (!os.is_open())
    return -1;

  constexpr bool is_float =
      (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16> ||
       std::is_same_v<T, double> || std::is_same_v<T, float>);
  constexpr unsigned ele_width = is_float ? 11 : 8;
  do {
    T *tmp = data;
    for (int h = 0; h < height; h++) {
      if (h == 0) {
        for (int w = 0; w < width; w++) {
          os << setw(ele_width) << static_cast<type>(w);
        }
        os << "\n";
      }
      for (int w = 0; w < width; w++) {
        if constexpr (std::is_same_v<T, half> ||
                             std::is_same_v<T, nv_bfloat16> ||
                             std::is_same_v<T, double> ||
                             std::is_same_v<T, float>) {
          // // 设置浮点输出占据位数不超过6位，考虑正负符号
          os << setw(ele_width) << fixed
             << setprecision(6 - ((float)tmp[w] < 0 ? 1 : 0))
             << static_cast<type>(tmp[w]);
        } else {
          os << setw(ele_width) << static_cast<type>(tmp[w]);
        }
      }
      tmp += stride;
      os << "\n";
    }

    ret = 0;
  } while (0);

  os.close();
  // fclose(fp);
  printf("printdata2txt %s finished.**********************\n", path);
  return ret;
}
