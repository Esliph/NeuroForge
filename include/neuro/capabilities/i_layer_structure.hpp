#pragma once

namespace neuro {

  class ILayerStructure {
   public:
    virtual ~ILayerStructure() = default;

    virtual void clear() = 0;

    virtual void reshape(size_t newInputSize, size_t newOutputSize) = 0;

    virtual size_t inputSize() const = 0;
    virtual size_t outputSize() const = 0;

    virtual bool validateInternalShape() = 0;
  };

} // namespace neuro
