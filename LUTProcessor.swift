import Metal
import CoreImage
import MetalKit
import Foundation

enum LUTError: Error {
    case invalidSize(expected: Int, got: Int)
    case invalidFormat(String)
    case textureCreationFailed
    case loadingFailed(String)
}

enum LUTSize: Int {
    case size17 = 17
    case size33 = 33
    
    var totalEntries: Int {
        return self.rawValue * self.rawValue * self.rawValue  // Ceci est le nombre de points 3D
    }
    
    var rgbaSize: Int {
        return totalEntries * 4  // Pour le format RGBA
    }
    
    var rgbSize: Int {
        return totalEntries * 3  // Pour le format RGB (fichier .cube)
    }
}

enum ColorSpace {
    case rec709
    case rec2020
    case p3
    case aces
    case logC
    case sLog3
}

struct ColorTransform {
    let sourceSpace: ColorSpace
    let targetSpace: ColorSpace
    let lutPath: String?
    
    // Transformations prédéfinies
    static let logCToRec709 = ColorTransform(sourceSpace: .logC, targetSpace: .rec709, lutPath: "LogC_to_Rec709")
    static let sLog3ToRec709 = ColorTransform(sourceSpace: .sLog3, targetSpace: .rec709, lutPath: "SLog3_to_Rec709")
    static let acesToRec709 = ColorTransform(sourceSpace: .aces, targetSpace: .rec709, lutPath: "ACES_to_Rec709")
}

private struct LUTMetadata {
    let title: String
    let size: Int
    let description: String
    let inputRange: ClosedRange<Float>
    let outputRange: ClosedRange<Float>
}

// Ajouter cette structure pour les paramètres d'espace colorimétrique
struct ColorSpaceParameters {
    var sourceGamma: SIMD3<Float>
    var targetGamma: SIMD3<Float>
    var sourceMatrix: matrix_float3x3
    var targetMatrix: matrix_float3x3
}

class LUTProcessor {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue?
    private var lutCache: [String: MTLTexture] = [:]
    private var pipelineState: MTLComputePipelineState?
    private let fileManager = FileManager.default
    
    init(device: MTLDevice) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()
        
        // Créer le pipeline state une seule fois
        if let library = device.makeDefaultLibrary(),
           let kernelFunction = library.makeFunction(name: "applyLUT") {
            do {
                pipelineState = try device.makeComputePipelineState(function: kernelFunction)
            } catch {
                print("Erreur lors de la création du pipeline state: \(error)")
            }
        }
    }
    
    // Obtenir la texture LUT pour une LUT donnée
    func getLUTTexture(for lut: InputLUT) -> MTLTexture? {
        switch lut {
        case .none:
            // Retourner une LUT identité pour le cas .none
            return getIdentityLUTTexture()
            
        case .custom(let customData):
            // LUT personnalisée
            return getCustomLUTTexture(customData: customData)
            
        default:
            // LUT intégrée
            let lutName = lut.getLUTFileName()
            
            // Vérifier si la texture est déjà en cache
            if let cachedTexture = lutCache[lutName] {
                return cachedTexture
            }
            
            // Sinon, charger la texture
            if let texture = loadBuiltInLUTTexture(named: lutName) {
                lutCache[lutName] = texture
                return texture
            }
        }
        
        return nil
    }
    
    // Obtenir une texture LUT personnalisée
    private func getCustomLUTTexture(customData: LUTCustomData) -> MTLTexture? {
        let cacheKey = "custom_\(customData.id.uuidString)"
        
        // Vérifier si la texture est déjà en cache
        if let cachedTexture = lutCache[cacheKey] {
            print("✅ LUT personnalisée trouvée en cache: \(customData.name)")
            return cachedTexture
        }
        
        // Vérifier que le fichier existe
        let filePath = customData.filePath
        guard fileManager.fileExists(atPath: filePath) else {
            print("❌ Fichier LUT personnalisée introuvable: \(filePath)")
            return nil
        }
        
        // Charger la LUT depuis le fichier
        do {
            let fileURL = URL(fileURLWithPath: filePath)
            let data = try String(contentsOf: fileURL)
            
            if let texture = createLUTTexture(from: data) {
                // Mettre en cache
                lutCache[cacheKey] = texture
                print("✅ LUT personnalisée chargée et mise en cache: \(customData.name)")
                return texture
            }
        } catch {
            print("❌ Erreur lecture fichier LUT: \(error)")
        }
        
        return nil
    }
    
    // Charger une LUT intégrée
    private func loadBuiltInLUTTexture(named name: String) -> MTLTexture? {
        guard let url = Bundle.main.url(forResource: name, withExtension: "cube") else {
            print("❌ Fichier LUT intégrée introuvable: \(name).cube")
            return nil
        }
        
        do {
            let data = try String(contentsOf: url)
            return createLUTTexture(from: data)
        } catch {
            print("❌ Erreur lecture fichier LUT: \(error)")
            return nil
        }
    }
    
    // Créer une texture LUT à partir des données du fichier
    private func createLUTTexture(from data: String) -> MTLTexture? {
        let lines = data.components(separatedBy: .newlines)
        var size = 33 // Taille par défaut
        var values: [Float] = []
        
        // Parser le fichier .cube
        for line in lines {
            let trimmedLine = line.trimmingCharacters(in: .whitespaces)
            
            if trimmedLine.isEmpty || trimmedLine.hasPrefix("#") {
                continue
            }
            
            if trimmedLine.hasPrefix("LUT_3D_SIZE") {
                if let sizeStr = trimmedLine.components(separatedBy: .whitespaces).last,
                   let lutSize = Int(sizeStr) {
                    size = lutSize
                }
                continue
            }
            
            let components = trimmedLine.components(separatedBy: .whitespaces)
                                     .filter { !$0.isEmpty }
            
            if components.count == 3,
               let r = Float(components[0]),
               let g = Float(components[1]),
               let b = Float(components[2]) {
                values.append(r)
                values.append(g)
                values.append(b)
                values.append(1.0) // Alpha
            }
        }
        
        // Vérifier que les données sont valides
        let expectedSize = size * size * size * 4 // RGBA
        if values.count != expectedSize {
            print("⚠️ Taille de données différente de l'attendu")
            print("📊 Obtenu: \(values.count) valeurs")
            print("📊 Attendu: \(expectedSize) valeurs")
            // Continuer quand même
        }
        
        // Créer la texture
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.textureType = .type3D
        textureDescriptor.pixelFormat = .rgba32Float
        textureDescriptor.width = size
        textureDescriptor.height = size
        textureDescriptor.depth = size
        textureDescriptor.usage = [.shaderRead]
        
        guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
            print("❌ Échec création texture LUT")
            return nil
        }
        
        // Copier les données dans la texture
        let region = MTLRegion(
            origin: MTLOrigin(x: 0, y: 0, z: 0),
            size: MTLSize(width: size, height: size, depth: size)
        )
        
        texture.replace(
            region: region,
            mipmapLevel: 0,
            slice: 0,
            withBytes: values,
            bytesPerRow: size * 4 * MemoryLayout<Float>.size,
            bytesPerImage: size * size * 4 * MemoryLayout<Float>.size
        )
        
        print("✅ LUT chargée avec succès, taille=\(size)")
        return texture
    }
    
    // Créer une LUT identité pour le cas .none
    private func getIdentityLUTTexture() -> MTLTexture? {
        let cacheKey = "identity_lut"
        
        // Vérifier si déjà en cache
        if let cachedTexture = lutCache[cacheKey] {
            return cachedTexture
        }
        
        // Créer une LUT identité 33x33x33
        let size = 33
        var values: [Float] = []
        
        for z in 0..<size {
            for y in 0..<size {
                for x in 0..<size {
                    let r = Float(x) / Float(size - 1)
                    let g = Float(y) / Float(size - 1)
                    let b = Float(z) / Float(size - 1)
                    
                    values.append(r)
                    values.append(g)
                    values.append(b)
                    values.append(1.0) // Alpha
                }
            }
        }
        
        // Créer la texture
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.textureType = .type3D
        textureDescriptor.pixelFormat = .rgba32Float
        textureDescriptor.width = size
        textureDescriptor.height = size
        textureDescriptor.depth = size
        textureDescriptor.usage = [.shaderRead]
        
        guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
            print("❌ Échec création texture LUT identité")
            return nil
        }
        
        // Copier les données dans la texture
        let region = MTLRegion(
            origin: MTLOrigin(x: 0, y: 0, z: 0),
            size: MTLSize(width: size, height: size, depth: size)
        )
        
        texture.replace(
            region: region,
            mipmapLevel: 0,
            slice: 0,
            withBytes: values,
            bytesPerRow: size * 4 * MemoryLayout<Float>.size,
            bytesPerImage: size * size * 4 * MemoryLayout<Float>.size
        )
        
        // Mettre en cache
        lutCache[cacheKey] = texture
        print("✅ LUT identité créée et mise en cache")
        return texture
    }
    
    // Vider le cache
    func clearCache() {
        lutCache.removeAll()
        print("✅ Cache LUT vidé")
    }
    
    // Recharger les LUTs personnalisées
    func reloadCustomLUTs() {
        // Vider le cache des LUTs personnalisées
        let customKeys = lutCache.keys.filter { $0.hasPrefix("custom_") }
        for key in customKeys {
            lutCache.removeValue(forKey: key)
        }
        
        print("✅ Cache des LUTs personnalisées vidé")
    }
    
    // Appliquer une LUT à un buffer
    func applyLUT(to sourceBuffer: CVPixelBuffer, lut: InputLUT) -> CVPixelBuffer? {
        guard lut != .none else { return sourceBuffer }
        
        guard let lutTexture = getLUTTexture(for: lut) else {
            print("❌ Texture LUT non disponible pour: \(lut.displayName)")
            return sourceBuffer
        }
        
        // Créer un nouveau buffer pour le résultat
        var destinationBuffer: CVPixelBuffer?
        let width = CVPixelBufferGetWidth(sourceBuffer)
        let height = CVPixelBufferGetHeight(sourceBuffer)
        
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width, height,
            CVPixelBufferGetPixelFormatType(sourceBuffer),
            [
                kCVPixelBufferMetalCompatibilityKey: true,
                kCVPixelBufferIOSurfacePropertiesKey: [:]
            ] as CFDictionary,
            &destinationBuffer
        )
        
        guard status == kCVReturnSuccess, let destinationBuffer = destinationBuffer else {
            print("❌ Échec création buffer de destination")
            return sourceBuffer
        }
        
        // Créer les textures Metal
        var sourceTexture: MTLTexture?
        var destinationTexture: MTLTexture?
        
        var textureCache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
        
        guard let cache = textureCache else {
            print("❌ Échec création texture cache")
            return sourceBuffer
        }
        
        // Créer la texture source
        var cvSourceTexture: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            cache,
            sourceBuffer,
            nil,
            .bgra8Unorm,
            width,
            height,
            0,
            &cvSourceTexture
        )
        
        if let cvTexture = cvSourceTexture {
            sourceTexture = CVMetalTextureGetTexture(cvTexture)
        }
        
        // Créer la texture destination
        var cvDestTexture: CVMetalTexture?
        CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            cache,
            destinationBuffer,
            nil,
            .bgra8Unorm,
            width,
            height,
            0,
            &cvDestTexture
        )
        
        if let cvTexture = cvDestTexture {
            destinationTexture = CVMetalTextureGetTexture(cvTexture)
        }
        
        guard let srcTexture = sourceTexture, let destTexture = destinationTexture else {
            print("❌ Échec création textures Metal")
            return sourceBuffer
        }
        
        // Créer un command buffer
        guard let commandBuffer = commandQueue?.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder(),
              let pipelineState = pipelineState else {
            print("❌ Échec création command buffer")
            return sourceBuffer
        }
        
        // Configurer l'encoder
        computeEncoder.setComputePipelineState(pipelineState)
        computeEncoder.setTexture(srcTexture, index: 0)
        computeEncoder.setTexture(lutTexture, index: 1)
        computeEncoder.setTexture(destTexture, index: 2)
        
        // Dispatcher
        let threadGroupSize = MTLSizeMake(16, 16, 1)
        let threadGroups = MTLSizeMake(
            (width + threadGroupSize.width - 1) / threadGroupSize.width,
            (height + threadGroupSize.height - 1) / threadGroupSize.height,
            1
        )
        
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
        
        // Exécuter
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return destinationBuffer
    }
    
    // Importer une LUT personnalisée (cette méthode est un pont vers CustomLUTManager)
    func importCustomLUT(from url: URL) -> Bool {
        // Cette méthode est un pont vers CustomLUTManager
        // Elle est implémentée ici pour maintenir la compatibilité avec le code existant
        print("⚠️ Cette méthode est un pont vers CustomLUTManager")
        print("⚠️ Utilisez CustomLUTManager.importLUT(from:) à la place")
        
        // Vider le cache pour forcer le rechargement
        clearCache()
        
        return true
    }
    
    // Obtenir la liste des LUTs personnalisées
    func getCustomLUTs() -> [CustomLUT] {
        // Cette méthode est un pont vers CustomLUTManager
        print("⚠️ Cette méthode est un pont vers CustomLUTManager")
        print("⚠️ Utilisez CustomLUTManager.customLUTs à la place")
        
        return []
    }
    
    // Supprimer une LUT personnalisée
    func deleteCustomLUT(named name: String) -> Bool {
        // Cette méthode est un pont vers CustomLUTManager
        print("⚠️ Cette méthode est un pont vers CustomLUTManager")
        print("⚠️ Utilisez CustomLUTManager.deleteLUT(_:) à la place")
        
        // Vider le cache pour forcer le rechargement
        clearCache()
        
        return true
    }
}

// Ajout de la structure pour gérer les chemins des LUTs
private struct LUTResources {
    static let lutsFolderName = "LUTs"
    
    // Métadonnées pour chaque LUT
    static let metadata: [InputLUT: LUTMetadata] = [
        .arriLogC4: LUTMetadata(
            title: "ARRI LogC4",
            size: 33,
            description: "ARRI LogC4 to Rec.709",
            inputRange: 0.0...1.0,
            outputRange: 0.0...1.0
        ),
        .bmdFilmGen3: LUTMetadata(
            title: "Blackmagic 4.6K Film Gen3",
            size: 33,
            description: "Blackmagic 4.6K Film Gen3 to Rec.709",
            inputRange: 0.0...1.0,
            outputRange: 0.0...1.0
        ),
        .bmdFilmGen4: LUTMetadata(
            title: "BMPCC 4K Gen4",
            size: 33,
            description: "BMPCC 4K Gen4 to Rec.709",
            inputRange: 0.0...1.0,
            outputRange: 0.0...1.0
        ),
        .bmdFilmGen5: LUTMetadata(
            title: "BMPCC 4K Gen5",
            size: 33,
            description: "BMPCC 4K Gen5 to Rec.709",
            inputRange: 0.0...1.0,
            outputRange: 0.0...1.0
        ),
        .bmpcc6KGen4: LUTMetadata(
            title: "BMPCC 6K Gen4",
            size: 33,
            description: "BMPCC 6K Gen4 to Rec.709",
            inputRange: 0.0...1.0,
            outputRange: 0.0...1.0
        ),
        .bmpcc6KGen5: LUTMetadata(
            title: "BMPCC 6K Gen5",
            size: 33,
            description: "BMPCC 6K Gen5 to Rec.709",
            inputRange: 0.0...1.0,
            outputRange: 0.0...1.0
        ),
        .canonCLog: LUTMetadata(
            title: "Canon C-Log",
            size: 33,
            description: "Canon C-Log to Rec.709",
            inputRange: 0.0...1.0,
            outputRange: 0.0...1.0
        ),
        .canonCLog2: LUTMetadata(
            title: "Canon C-Log2",
            size: 33,
            description: "Canon C-Log2 to Rec.709",
            inputRange: 0.0...1.0,
            outputRange: 0.0...1.0
        ),
        .canonCLog3: LUTMetadata(
            title: "Canon C-Log3",
            size: 33,
            description: "Canon C-Log3 to Rec.709",
            inputRange: 0.0...1.0,
            outputRange: 0.0...1.0
        ),
        .djiDLogM: LUTMetadata(
            title: "DJI D-Log M",
            size: 33,
            description: "DJI D-Log M to Rec.709",
            inputRange: 0.0...1.0,
            outputRange: 0.0...1.0
        ),
        .djiDLog: LUTMetadata(
            title: "DJI D-Log",
            size: 33,
            description: "DJI D-Log to Rec.709",
            inputRange: 0.0...1.0,
            outputRange: 0.0...1.0
        ),
        .fujifilmFLog: LUTMetadata(
            title: "Fujifilm F-Log",
            size: 33,
            description: "Fujifilm F-Log to Rec.709",
            inputRange: 0.0...1.0,
            outputRange: 0.0...1.0
        ),
        .nikonNLog: LUTMetadata(
            title: "Nikon N-Log",
            size: 33,
            description: "Nikon N-Log to Rec.709",
            inputRange: 0.0...1.0,
            outputRange: 0.0...1.0
        ),
        .panasonicVLog: LUTMetadata(
            title: "Panasonic V-Log",
            size: 33,
            description: "Panasonic V-Log to Rec.709",
            inputRange: 0.0...1.0,
            outputRange: 0.0...1.0
        ),
        .redLogFilm: LUTMetadata(
            title: "RED RedLogFilm",
            size: 33,
            description: "RED RedLogFilm to Rec.709",
            inputRange: 0.0...1.0,
            outputRange: 0.0...1.0
        ),
        .sonySLog2: LUTMetadata(
            title: "Sony S-Log2",
            size: 33,
            description: "Sony S-Log2 to Rec.709",
            inputRange: 0.0...1.0,
            outputRange: 0.0...1.0
        ),
        .sonySLog3: LUTMetadata(
            title: "Sony S-Log3",
            size: 33,
            description: "Sony S-Log3 to Rec.709",
            inputRange: 0.0...1.0,
            outputRange: 0.0...1.0
        ),
        .none: LUTMetadata(
            title: "Rec.709",
            size: 33,
            description: "Standard Rec.709 color space",
            inputRange: 0.0...1.0,
            outputRange: 0.0...1.0
        )
    ]
    
    static func path(for lut: InputLUT) -> String? {
        let fileName: String
        switch lut {
        case .none:
            return nil  // Pas de fichier nécessaire pour Rec.709
        case .arriLogC4:
            fileName = "ARRI LogC4"
        case .bmdFilmGen3:
            fileName = "Blackmagic 4.6K Film Gen3"
        case .bmdFilmGen4:
            fileName = "BMPCC 4K Gen4"
        case .bmdFilmGen5:
            fileName = "BMPCC 4K Gen5"
        case .bmpcc6KGen4:
            fileName = "BMPCC 6K Gen4"
        case .bmpcc6KGen5:
            fileName = "BMPCC 6K Gen5"
        case .canonCLog:
            fileName = "Canon C-Log"
        case .canonCLog2:
            fileName = "Canon C-Log2"
        case .canonCLog3:
            fileName = "Canon C-Log3"
        case .djiDLog:
            fileName = "DJI D-Log"
        case .djiDLogM:
            fileName = "DJI D-Log M"
        case .fujifilmFLog:
            fileName = "Fujifilm F-Log"
        case .nikonNLog:
            fileName = "Nikon N-Log"
        case .panasonicVLog:
            fileName = "Panasonic V-Log"
        case .redLogFilm:
            fileName = "RED RedLogFilm"
        case .sonySLog2:
            fileName = "Sony S-Log2"
        case .sonySLog3:
            fileName = "Sony S-Log3"
        case .custom(let customLUT):
            return customLUT.filePath
        }
        
        // Essayer de trouver le fichier LUT dans le bundle
        if let path = Bundle.main.path(forResource: fileName, ofType: "cube") {
            print("✅ Found LUT at: \(path)")
            return path
        }
        
        print("❌ LUT file not found: \(fileName).cube")
        return nil
    }
} 
