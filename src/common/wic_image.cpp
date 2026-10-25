#include "common/wic_image.h"

#include "common/win_util.h"

#include <wincodec.h>

namespace nr {

BgraImage LoadBgraImage(const std::filesystem::path& path) {
  Microsoft::WRL::ComPtr<IWICImagingFactory> factory;
  Check(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)),
        "CoCreateInstance(WICImagingFactory)");
  Microsoft::WRL::ComPtr<IWICBitmapDecoder> decoder;
  Check(factory->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand,
                                           &decoder),
        "CreateDecoderFromFilename");
  Microsoft::WRL::ComPtr<IWICBitmapFrameDecode> frame;
  Check(decoder->GetFrame(0, &frame), "GetFrame");
  Microsoft::WRL::ComPtr<IWICFormatConverter> converter;
  Check(factory->CreateFormatConverter(&converter), "CreateFormatConverter");
  Check(converter->Initialize(frame.Get(), GUID_WICPixelFormat32bppBGRA, WICBitmapDitherTypeNone, nullptr, 0,
                              WICBitmapPaletteTypeCustom),
        "IWICFormatConverter::Initialize");

  BgraImage image;
  Check(converter->GetSize(&image.width, &image.height), "GetSize");
  image.pixels.resize(static_cast<size_t>(image.width) * image.height * 4);
  Check(converter->CopyPixels(nullptr, image.width * 4, static_cast<UINT>(image.pixels.size()), image.pixels.data()),
        "CopyPixels");
  return image;
}

Microsoft::WRL::ComPtr<ID3D11Texture2D> CreateTexture(ID3D11Device* device, const BgraImage& image) {
  D3D11_TEXTURE2D_DESC td{};
  td.Width = image.width;
  td.Height = image.height;
  td.MipLevels = 1;
  td.ArraySize = 1;
  td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
  td.SampleDesc.Count = 1;
  td.Usage = D3D11_USAGE_DEFAULT;
  td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
  D3D11_SUBRESOURCE_DATA data{image.pixels.data(), image.width * 4, 0};
  Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
  Check(device->CreateTexture2D(&td, &data, &texture), "CreateTexture2D(图片)");
  return texture;
}

}  // namespace nr
