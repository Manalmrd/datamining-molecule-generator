import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

function App() {
  const [properties, setProperties] = useState({
    Molecular_Weight_norm: 0.5,
    XLogP3_norm: 0.5,
    Topological_Polar_Surface_Area_norm: 0.5,
    Rotatable_Bond_Count_norm: 0.5,
    Hydrogen_Bond_Acceptor_Count_norm: 0.5
  });

  const [generatedMolecule, setGeneratedMolecule] = useState('');
  const [moleculeName, setMoleculeName] = useState('');
  const [predictedProperties, setPredictedProperties] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [moleculeImage, setMoleculeImage] = useState(null);
  const [moleculeSmiles, setMoleculeSmiles] = useState('');
  const [moleculeDescription, setMoleculeDescription] = useState('');
  const [moleculeAnalysis, setMoleculeAnalysis] = useState(null);
  const [moleculeId, setMoleculeId] = useState('');
  const [is3DLoading, setIs3DLoading] = useState(false);
  const [showSMILESTooltip, setShowSMILESTooltip] = useState(false);
  const [viewMode, setViewMode] = useState('split');
  const [isGeneratingPDF, setIsGeneratingPDF] = useState(false);

  // Références pour Three.js
  const containerRef3D = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const animationRef = useRef(null);

  // Référence pour capture PDF
  const reportRef = useRef(null);

  const propertyLabels = {
    Molecular_Weight_norm: 'Poids moléculaire',
    XLogP3_norm: 'XLogP3 (Solubilité)',
    Topological_Polar_Surface_Area_norm: 'Surface polaire topologique',
    Rotatable_Bond_Count_norm: 'Liaisons rotables (Flexibilité)',
    Hydrogen_Bond_Acceptor_Count_norm: 'Accepteurs de liaison H'
  };

  const propertyTooltips = {
    Molecular_Weight_norm: 'Masse totale de la molécule en g/mol',
    XLogP3_norm: 'LogP prédit - mesure la lipophilie (solubilité)',
    Topological_Polar_Surface_Area_norm: 'Surface accessible aux atomes polaires en Å²',
    Rotatable_Bond_Count_norm: 'Nombre de liaisons simples non-cycliques',
    Hydrogen_Bond_Acceptor_Count_norm: 'Atomes d\'oxygène et d\'azote capables d\'accepter des liaisons H'
  };

  const propertyUnits = {
    Molecular_Weight_norm: 'Masse totale des atomes dans la molécule (g/mol).',
    XLogP3_norm: 'Tendance à se dissoudre dans les graisses ou l\'eau.',
    Topological_Polar_Surface_Area_norm: 'Surface attirant l\'eau qui affecte le passage des membranes (Å²).',
    Rotatable_Bond_Count_norm: 'Nombre de liaisons qui peuvent tourner librement.',
    Hydrogen_Bond_Acceptor_Count_norm: 'Points d\'attache pour les liaisons hydrogène.'
  };

  // Nettoyage Three.js
  useEffect(() => {
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (controlsRef.current) {
        controlsRef.current.dispose();
      }
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
    };
  }, []);

  // Redimensionnement du renderer
  useEffect(() => {
    const handleResize = () => {
      if (rendererRef.current && containerRef3D.current) {
        const width = containerRef3D.current.clientWidth;
        const height = containerRef3D.current.clientHeight;

        if (cameraRef.current) {
          cameraRef.current.aspect = width / height;
          cameraRef.current.updateProjectionMatrix();
        }

        rendererRef.current.setSize(width, height);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Fonction pour générer le rapport PDF
  const generatePDFReport = async () => {
    if (!moleculeAnalysis || !moleculeImage) {
      alert('Veuillez générer une molécule avant de créer un rapport');
      return;
    }

    setIsGeneratingPDF(true);

    try {
      const pdf = new jsPDF('portrait', 'mm', 'a4');
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 15;
      const contentWidth = pageWidth - 2 * margin;

      // Titre du rapport
      pdf.setFontSize(20);
      pdf.setFont('helvetica', 'bold');
      pdf.text('Rapport de Molécule Générée', pageWidth / 2, margin, { align: 'center' });

      // Date et heure
      pdf.setFontSize(10);
      pdf.setFont('helvetica', 'normal');
      const now = new Date();
      const dateStr = now.toLocaleDateString('fr-FR');
      const timeStr = now.toLocaleTimeString('fr-FR');
      pdf.text(`Généré le: ${dateStr} à ${timeStr}`, pageWidth / 2, margin + 8, { align: 'center' });

      let yPos = margin + 20;

      // Informations de base
      pdf.setFontSize(14);
      pdf.setFont('helvetica', 'bold');
      pdf.text('Informations de la Molécule', margin, yPos);
      yPos += 8;

      pdf.setFontSize(11);
      pdf.setFont('helvetica', 'normal');
      pdf.text(`Nom: ${moleculeName}`, margin, yPos);
      yPos += 6;

      if (moleculeDescription) {
        const descriptionLines = pdf.splitTextToSize(moleculeDescription, contentWidth);
        pdf.text(descriptionLines, margin, yPos);
        yPos += descriptionLines.length * 5 + 4;
      }

      if (moleculeAnalysis?.properties?.formula) {
        pdf.text(`Formule: ${moleculeAnalysis.properties.formula}`, margin, yPos);
        yPos += 6;
      }

      if (moleculeSmiles) {
        const smilesLines = pdf.splitTextToSize(`SMILES: ${moleculeSmiles}`, contentWidth);
        pdf.text(smilesLines, margin, yPos);
        yPos += smilesLines.length * 5 + 4;
      }

      yPos += 5;

      // Propriétés cibles
      pdf.setFontSize(14);
      pdf.setFont('helvetica', 'bold');
      pdf.text('Propriétés Cibles', margin, yPos);
      yPos += 8;

      pdf.setFontSize(11);
      pdf.setFont('helvetica', 'normal');
      Object.entries(properties).forEach(([key, value]) => {
        pdf.text(`${propertyLabels[key]}: ${value.toFixed(2)}`, margin, yPos);
        yPos += 6;
      });

      yPos += 5;

      // Propriétés prédites
      if (predictedProperties.length > 0) {
        pdf.setFontSize(14);
        pdf.setFont('helvetica', 'bold');
        pdf.text('Propriétés Prédites', margin, yPos);
        yPos += 8;

        pdf.setFontSize(11);
        pdf.setFont('helvetica', 'normal');
        predictedProperties.forEach((prop, index) => {
          const target = properties[Object.keys(properties)[index]]?.toFixed(2);
          pdf.text(`${prop} (Cible: ${target})`, margin, yPos);
          yPos += 6;
        });

        yPos += 5;
      }

      // Statistiques
      if (moleculeAnalysis) {
        pdf.setFontSize(14);
        pdf.setFont('helvetica', 'bold');
        pdf.text('Statistiques Structurales', margin, yPos);
        yPos += 8;

        pdf.setFontSize(11);
        pdf.setFont('helvetica', 'normal');
        const stats = [
          `Atomes totaux: ${moleculeAnalysis.num_atoms_3d || 0}`,
          `Atomes lourds: ${moleculeAnalysis.properties?.heavy_atoms || 0}`,
          `Liaisons: ${moleculeAnalysis.bonds_3d?.length || 0}`,
          `Cycles: ${moleculeAnalysis.properties?.num_rings || 0}`,
          `Liaisons rotatives: ${moleculeAnalysis.properties?.rotatable_bonds || 0}`,
          `Accepteurs H: ${moleculeAnalysis.properties?.h_bond_acceptors || 0}`,
          `Donneurs H: ${moleculeAnalysis.properties?.h_bond_donors || 0}`
        ];

        stats.forEach(stat => {
          pdf.text(stat, margin, yPos);
          yPos += 6;
        });

        yPos += 10;
      }

      // Vérifier si on a besoin d'une nouvelle page
      if (yPos > pageHeight - 100) {
        pdf.addPage();
        yPos = margin;
      }

      // Image 2D
      pdf.setFontSize(14);
      pdf.setFont('helvetica', 'bold');
      pdf.text('Structure 2D', pageWidth / 2, yPos, { align: 'center' });
      yPos += 8;

      try {
        // Convertir l'image base64 en blob
        const img = new Image();
        img.src = moleculeImage;

        await new Promise((resolve) => {
          img.onload = resolve;
        });

        // Créer un canvas pour redimensionner l'image
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const maxWidth = 150;
        const maxHeight = 150;
        let width = img.width;
        let height = img.height;

        if (width > height) {
          if (width > maxWidth) {
            height *= maxWidth / width;
            width = maxWidth;
          }
        } else {
          if (height > maxHeight) {
            width *= maxHeight / height;
            height = maxHeight;
          }
        }

        canvas.width = width;
        canvas.height = height;
        ctx.drawImage(img, 0, 0, width, height);

        const resizedImage = canvas.toDataURL('image/png');
        pdf.addImage(resizedImage, 'PNG', pageWidth / 2 - width / 4, yPos, width / 2, height / 2);
        yPos += height / 2 + 20;
      } catch (error) {
        console.error('Erreur lors du chargement de l\'image 2D:', error);
        pdf.text('Image 2D non disponible', margin, yPos);
        yPos += 10;
      }

      // Nouvelle page pour la 3D si nécessaire
      if (yPos > pageHeight - 100) {
        pdf.addPage();
        yPos = margin;
      }

      // Capture 3D
      if (rendererRef.current && !is3DLoading) {
        pdf.setFontSize(14);
        pdf.setFont('helvetica', 'bold');
        pdf.text('Structure 3D', pageWidth / 2, yPos, { align: 'center' });
        yPos += 8;

        try {
          // Capturer le rendu 3D
          const dataURL = rendererRef.current.domElement.toDataURL('image/png');

          const img3D = new Image();
          img3D.src = dataURL;

          await new Promise((resolve) => {
            img3D.onload = resolve;
          });

          // Redimensionner l'image 3D
          const canvas3D = document.createElement('canvas');
          const ctx3D = canvas3D.getContext('2d');
          const maxWidth3D = 150;
          const maxHeight3D = 150;
          let width3D = img3D.width;
          let height3D = img3D.height;

          if (width3D > height3D) {
            if (width3D > maxWidth3D) {
              height3D *= maxWidth3D / width3D;
              width3D = maxWidth3D;
            }
          } else {
            if (height3D > maxHeight3D) {
              width3D *= maxHeight3D / height3D;
              height3D = maxHeight3D;
            }
          }

          canvas3D.width = width3D;
          canvas3D.height = height3D;
          ctx3D.drawImage(img3D, 0, 0, width3D, height3D);

          const resizedImage3D = canvas3D.toDataURL('image/png');
          pdf.addImage(resizedImage3D, 'PNG', pageWidth / 2 - width3D / 4, yPos, width3D / 2, height3D / 2);
          yPos += height3D / 2 + 20;
        } catch (error) {
          console.error('Erreur lors de la capture 3D:', error);
          pdf.text('Vue 3D non disponible', margin, yPos);
          yPos += 10;
        }
      }

      // Pied de page
      pdf.setFontSize(10);
      pdf.setFont('helvetica', 'italic');
      pdf.text('Généré avec Générateur de Molécules - www.votresite.com', pageWidth / 2, pageHeight - 10, { align: 'center' });

      // Sauvegarder le PDF
      const fileName = `rapport_molecule_${moleculeId}_${now.getFullYear()}${(now.getMonth()+1).toString().padStart(2, '0')}${now.getDate().toString().padStart(2, '0')}_${now.getHours().toString().padStart(2, '0')}${now.getMinutes().toString().padStart(2, '0')}.pdf`;
      pdf.save(fileName);

    } catch (error) {
      console.error('Erreur lors de la génération du PDF:', error);
      alert('Erreur lors de la génération du rapport PDF');
    } finally {
      setIsGeneratingPDF(false);
    }
  };

  // Alternative: Générer le rapport à partir du HTML
  const generateHTMLReport = async () => {
    if (!reportRef.current) return;

    setIsGeneratingPDF(true);

    try {
      const element = reportRef.current;
      const canvas = await html2canvas(element, {
        scale: 2,
        useCORS: true,
        backgroundColor: '#ffffff'
      });

      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF('portrait', 'mm', 'a4');
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();

      // Calculer les dimensions pour ajuster l'image à la page
      const imgWidth = canvas.width;
      const imgHeight = canvas.height;
      const ratio = Math.min(pageWidth / imgWidth, pageHeight / imgHeight);
      const pdfWidth = imgWidth * ratio;
      const pdfHeight = imgHeight * ratio;

      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);

      const now = new Date();
      const fileName = `rapport_molecule_${moleculeId}_${now.getFullYear()}${(now.getMonth()+1).toString().padStart(2, '0')}${now.getDate().toString().padStart(2, '0')}_${now.getHours().toString().padStart(2, '0')}${now.getMinutes().toString().padStart(2, '0')}.pdf`;
      pdf.save(fileName);

    } catch (error) {
      console.error('Erreur lors de la génération du rapport HTML:', error);
      alert('Erreur lors de la génération du rapport');
    } finally {
      setIsGeneratingPDF(false);
    }
  };

  // Rendu 3D
  const renderMolecule3D = (analysis) => {
    if (!analysis || !analysis.atoms_3d || !analysis.bonds_3d) {
      console.log('❌ Pas de données 3D disponibles');
      setIs3DLoading(false);
      return;
    }

    setIs3DLoading(true);
    console.log('🎨 Rendu 3D avec:', analysis.atoms_3d.length, 'atomes');

    // Nettoyer le conteneur
    if (containerRef3D.current) {
      containerRef3D.current.innerHTML = '';
    }

    // Initialiser Three.js
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a1a);

    const camera = new THREE.PerspectiveCamera(
      45,
      containerRef3D.current.clientWidth / containerRef3D.current.clientHeight,
      0.1,
      1000
    );
    camera.position.z = 25;

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance",
      preserveDrawingBuffer: true // Important pour la capture PDF
    });
    renderer.setSize(
      containerRef3D.current.clientWidth,
      containerRef3D.current.clientHeight
    );
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    containerRef3D.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.rotateSpeed = 0.5;

    // Ajouter la lumière
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    // Couleurs des atomes (CPK colors)
    const atomColors = {
      'C': 0x909090,    // Gris
      'H': 0xFFFFFF,    // Blanc
      'O': 0xFF0000,    // Rouge
      'N': 0x0000FF,    // Bleu
      'S': 0xFFFF00,    // Jaune
      'F': 0x00FF00,    // Vert
      'Cl': 0x00FF00,   // Vert
      'Br': 0x8B0000,   // Rouge foncé
      'I': 0x9400D3,    // Violet
      'P': 0xFFA500,    // Orange
      'B': 0xFFB5C5,    // Rose
      'default': 0xFF69B4 // Rose vif
    };

    // Rayons des atomes
    const atomRadii = {
      'H': 0.3,
      'C': 0.7,
      'N': 0.65,
      'O': 0.6,
      'F': 0.5,
      'P': 1.0,
      'S': 1.0,
      'Cl': 1.0,
      'Br': 1.15,
      'I': 1.4,
      'default': 0.75
    };

    const atoms = analysis.atoms_3d;
    const bonds = analysis.bonds_3d;

    // Calculer le centre et la taille
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    atoms.forEach(atom => {
      minX = Math.min(minX, atom.x);
      maxX = Math.max(maxX, atom.x);
      minY = Math.min(minY, atom.y);
      maxY = Math.max(maxY, atom.y);
      minZ = Math.min(minZ, atom.z);
      maxZ = Math.max(maxZ, atom.z);
    });

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const centerZ = (minZ + maxZ) / 2;
    const size = Math.max(maxX - minX, maxY - minY, maxZ - minZ);

    // Ajuster la caméra en fonction de la taille
    if (size > 0) {
      camera.position.z = size * 2;
      camera.far = size * 10;
      camera.updateProjectionMatrix();
    }

    // Grouper la molécule
    const moleculeGroup = new THREE.Group();

    // Créer les atomes
    const atomSpheres = [];
    atoms.forEach(atom => {
      const radius = atomRadii[atom.symbol] || atomRadii.default;
      const color = atomColors[atom.symbol] || atomColors.default;

      const geometry = new THREE.SphereGeometry(radius, 24, 24);
      const material = new THREE.MeshPhongMaterial({
        color: color,
        shininess: 100,
        specular: 0x222222,
        transparent: true,
        opacity: 0.9
      });

      const sphere = new THREE.Mesh(geometry, material);

      // Position centrée
      sphere.position.set(
        atom.x - centerX,
        atom.y - centerY,
        atom.z - centerZ
      );

      sphere.castShadow = true;
      sphere.receiveShadow = true;

      moleculeGroup.add(sphere);
      atomSpheres.push(sphere);
    });

    // Créer les liaisons
    bonds.forEach(bond => {
      const atom1 = atoms[bond.from_atom];
      const atom2 = atoms[bond.to_atom];

      if (!atom1 || !atom2) return;

      const start = new THREE.Vector3(
        atom1.x - centerX,
        atom1.y - centerY,
        atom1.z - centerZ
      );

      const end = new THREE.Vector3(
        atom2.x - centerX,
        atom2.y - centerY,
        atom2.z - centerZ
      );

      const direction = new THREE.Vector3().subVectors(end, start);
      const length = direction.length();

      if (length > 0) {
        const geometry = new THREE.CylinderGeometry(0.08, 0.08, length, 8);
        const cylinder = new THREE.Mesh(
          geometry,
          new THREE.MeshPhongMaterial({
            color: 0xCCCCCC,
            shininess: 30
          })
        );

        cylinder.position.copy(start).add(end).multiplyScalar(0.5);
        cylinder.lookAt(end);
        cylinder.rotateX(Math.PI / 2);

        cylinder.castShadow = true;
        moleculeGroup.add(cylinder);
      }
    });

    scene.add(moleculeGroup);

    // Animation
    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };

    animate();

    // Stocker les références
    sceneRef.current = scene;
    rendererRef.current = renderer;
    cameraRef.current = camera;
    controlsRef.current = controls;
    setIs3DLoading(false);
  };

  // Générer la description
  const generateMoleculeDescription = (analysis) => {
    if (!analysis || !analysis.properties) return '';

    const props = analysis.properties;
    const descriptionParts = [];

    if (props.heavy_atoms > 0) {
      descriptionParts.push(`${props.heavy_atoms} atome${props.heavy_atoms > 1 ? 's' : ''} lourd${props.heavy_atoms > 1 ? 's' : ''}`);
    }

    if (props.num_rings > 0) {
      descriptionParts.push(`${props.num_rings} cycle${props.num_rings > 1 ? 's' : ''}`);
    }

    if (props.rotatable_bonds > 0) {
      descriptionParts.push(`${props.rotatable_bonds} liaison${props.rotatable_bonds > 1 ? 's' : ''} rotative${props.rotatable_bonds > 1 ? 's' : ''}`);
    }

    if (props.aromatic_atoms > 0) {
      descriptionParts.push(`${props.aromatic_atoms} atome${props.aromatic_atoms > 1 ? 's' : ''} aromatique${props.aromatic_atoms > 1 ? 's' : ''}`);
    }

    if (props.h_bond_acceptors > 0) {
      descriptionParts.push(`${props.h_bond_acceptors} accepteur${props.h_bond_acceptors > 1 ? 's' : ''} H`);
    }

    if (props.h_bond_donors > 0) {
      descriptionParts.push(`${props.h_bond_donors} donneur${props.h_bond_donors > 1 ? 's' : ''} H`);
    }

    let description = 'Molécule organique';
    if (descriptionParts.length > 0) {
      description += ' avec ' + descriptionParts.join(', ');
    }

    if (props.formula) {
      description += `. Formule: ${props.formula}`;
    }

    return description;
  };

  const handleSliderChange = (name, value) => {
    setProperties(prev => ({
      ...prev,
      [name]: parseFloat(value)
    }));
  };

  const copySMILES = () => {
    if (moleculeSmiles) {
      navigator.clipboard.writeText(moleculeSmiles);
      setShowSMILESTooltip(true);
      setTimeout(() => setShowSMILESTooltip(false), 2000);
    }
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    setMoleculeImage(null);
    setMoleculeDescription('');
    setPredictedProperties([]);
    setMoleculeAnalysis(null);
    setIs3DLoading(false);

    try {
      console.log('📤 Envoi des propriétés:', properties);

      const response = await fetch('http://localhost:5000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(properties)
      });

      const data = await response.json();
      console.log('📦 Données API complètes:', data);

      if (data.success) {
        setGeneratedMolecule(data.selfies);
        setMoleculeName(data.name || 'Molécule Générée');
        setMoleculeSmiles(data.smiles || '');
        setMoleculeId(data.molecule_id);

        // RÉCUPÉRER L'ANALYSE ET LES DONNÉES
        if (data.analysis) {
          console.log('🔬 Analyse reçue:', data.analysis);
          setMoleculeAnalysis(data.analysis);

          // RÉCUPÉRER L'IMAGE 2D
          if (data.analysis.image_2d) {
            const imageSrc = `data:image/png;base64,${data.analysis.image_2d}`;
            setMoleculeImage(imageSrc);
          }

          // GÉNÉRER LA DESCRIPTION
          const description = generateMoleculeDescription(data.analysis);
          setMoleculeDescription(description);

          // RENDRE LA STRUCTURE 3D
          if (data.analysis.atoms_3d && data.analysis.atoms_3d.length > 0) {
            setTimeout(() => {
              renderMolecule3D(data.analysis);
            }, 100);
          }
        }

        // RÉCUPÉRER LES PROPRIÉTÉS PRÉDITES
        if (data.predicted_properties_formatted && Array.isArray(data.predicted_properties_formatted)) {
          setPredictedProperties(data.predicted_properties_formatted);
        }

      } else {
        alert('Erreur lors de la génération: ' + (data.error || 'Erreur inconnue'));
      }
    } catch (error) {
      console.error("❌ Erreur complète:", error);
      alert("Erreur de connexion au serveur. Assurez-vous que l'API est active.");
    } finally {
      setIsLoading(false);
    }
  };

  // Contrôles 3D
  const handleZoomIn = () => {
    if (cameraRef.current) {
      cameraRef.current.position.z *= 0.8;
    }
  };

  const handleZoomOut = () => {
    if (cameraRef.current) {
      cameraRef.current.position.z *= 1.2;
    }
  };

  const handleResetView = () => {
    if (cameraRef.current && controlsRef.current) {
      cameraRef.current.position.set(0, 0, 25);
      cameraRef.current.lookAt(0, 0, 0);
      controlsRef.current.reset();
    }
  };

  const handleRotateLeft = () => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = true;
      controlsRef.current.autoRotateSpeed = -2;
    }
  };

  const handleRotateRight = () => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = true;
      controlsRef.current.autoRotateSpeed = 2;
    }
  };

  const handleStopRotation = () => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = false;
    }
  };

  const exportAsPNG = () => {
    if (rendererRef.current) {
      const image = rendererRef.current.domElement.toDataURL('image/png');
      const link = document.createElement('a');
      link.href = image;
      link.download = `molecule_${moleculeId}_3d.png`;
      link.click();
    }
  };

  // Composant pour le rapport caché
  const ReportContent = () => (
    <div ref={reportRef} style={{ display: 'none' }}>
      <div style={{ padding: '20px', backgroundColor: 'white', color: 'black' }}>
        <h1 style={{ textAlign: 'center', marginBottom: '10px' }}>
          Rapport de Molécule Générée
        </h1>
        <p style={{ textAlign: 'center', fontSize: '12px', marginBottom: '20px' }}>
          Généré le: {new Date().toLocaleDateString('fr-FR')} à {new Date().toLocaleTimeString('fr-FR')}
        </p>

        <div style={{ marginBottom: '20px' }}>
          <h2>Informations de la Molécule</h2>
          <p><strong>Nom:</strong> {moleculeName}</p>
          {moleculeDescription && <p><strong>Description:</strong> {moleculeDescription}</p>}
          {moleculeAnalysis?.properties?.formula && (
            <p><strong>Formule:</strong> {moleculeAnalysis.properties.formula}</p>
          )}
          {moleculeSmiles && <p><strong>SMILES:</strong> {moleculeSmiles}</p>}
        </div>

        <div style={{ marginBottom: '20px' }}>
          <h2>Propriétés Cibles</h2>
          {Object.entries(properties).map(([key, value]) => (
            <p key={key}><strong>{propertyLabels[key]}:</strong> {value.toFixed(2)}</p>
          ))}
        </div>

        {predictedProperties.length > 0 && (
          <div style={{ marginBottom: '20px' }}>
            <h2>Propriétés Prédites</h2>
            {predictedProperties.map((prop, index) => {
              const target = properties[Object.keys(properties)[index]]?.toFixed(2);
              return (
                <p key={index}><strong>{prop.split(':')[0]}:</strong> {prop.split(':')[1]} (Cible: {target})</p>
              );
            })}
          </div>
        )}

        {moleculeAnalysis && (
          <div style={{ marginBottom: '20px' }}>
            <h2>Statistiques Structurales</h2>
            <p><strong>Atomes totaux:</strong> {moleculeAnalysis.num_atoms_3d || 0}</p>
            <p><strong>Atomes lourds:</strong> {moleculeAnalysis.properties?.heavy_atoms || 0}</p>
            <p><strong>Liaisons:</strong> {moleculeAnalysis.bonds_3d?.length || 0}</p>
            <p><strong>Cycles:</strong> {moleculeAnalysis.properties?.num_rings || 0}</p>
            <p><strong>Liaisons rotatives:</strong> {moleculeAnalysis.properties?.rotatable_bonds || 0}</p>
            <p><strong>Accepteurs H:</strong> {moleculeAnalysis.properties?.h_bond_acceptors || 0}</p>
            <p><strong>Donneurs H:</strong> {moleculeAnalysis.properties?.h_bond_donors || 0}</p>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="app-container">
      <div className="max-w-7xl mx-auto">
        <header className="app-header">
          <div className="header-content">
            <div className="logo-container">
              <div className="logo-icon">
                <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
              </div>
              <div className="title-wrapper">
                <h1 className="main-title">Générateur de Molécules</h1>
                <p className="subtitle">Génération et visualisation interactive de molécules</p>
              </div>
            </div>

            {/* Bouton Export PDF */}
{generatedMolecule && (
  <div className="export-pdf-container">
    <button
      onClick={generatePDFReport}
      disabled={isGeneratingPDF}
      className="export-pdf-button"
      title="Générer un rapport PDF"
    >
      {isGeneratingPDF ? (
        <>
          <svg className="spinner" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          Génération PDF...
        </>
      ) : (
        <>
          {/* Utilisation de votre image icon_pdf.png */}
          <img
            src="/icon_pdf.png"
            alt="PDF Icon"
            className="icon-pdf"
            width="20"
            height="20"
          />
          Générer Rapport PDF
        </>
      )}
    </button>
  </div>
)}
          </div>
        </header>

        <div className="content-grid">
          {/* Panneau de contrôle */}
          <div className="properties-panel">
            <div className="panel-header">
              <h2 className="panel-title">Contrôle des Propriétés</h2>
              <p className="panel-description">Ajustez les valeurs (0.0 à 1.0)</p>
            </div>

            <div className="properties-list">
              {Object.entries(properties).map(([key, value]) => (
                <div key={key} className="property-item">
                  <div className="property-header">
                    <div className="property-label-group">
                      <h3
                        className="property-label"
                        title={propertyTooltips[key]}
                      >
                        {propertyLabels[key]}
                      </h3>
                      <span className="property-unit">{propertyUnits[key]}</span>
                    </div>
                    <div className="value-display">
                      <span className="value-number">{value.toFixed(2)}</span>
                    </div>
                  </div>

                  <div className="slider-container">
                    <div className="slider-scale">
                      <span className="scale-label">0.0</span>
                      <span className="scale-label">0.5</span>
                      <span className="scale-label">1.0</span>
                    </div>
                    <input
                      type="range"
                      name={key}
                      min="0"
                      max="1"
                      step="0.01"
                      value={value}
                      onChange={(e) => handleSliderChange(key, e.target.value)}
                      className="custom-slider"
                    />
                    <div className="slider-track">
                      <div
                        className="slider-progress"
                        style={{ width: `${value * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="generate-button-container">
              <button
                onClick={handleSubmit}
                disabled={isLoading}
                className="generate-button"
              >
                {isLoading ? (
                  <>
                    <svg className="spinner" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Génération en cours...
                  </>
                ) : (
                  <>
                    <svg className="generate-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                    </svg>
                    Générer la Molécule
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Panneau de visualisation */}
          <div className="visualization-panel">
            {generatedMolecule ? (
              <>
                {/* En-tête avec bouton PDF */}
                <div className="molecule-header">
                  <div className="molecule-header-top">
                    <h2 className="molecule-name">{moleculeName}</h2>
                  </div>

                  {moleculeDescription && (
                    <div className="molecule-description">
                      <p>{moleculeDescription}</p>
                    </div>
                  )}

                  {moleculeAnalysis?.properties?.formula && (
                    <div className="molecule-formula">
                      <strong>Formule:</strong> {moleculeAnalysis.properties.formula}
                    </div>
                  )}

                  {moleculeSmiles && (
                    <div className="smiles-display">
                      <div className="smiles-header">
                        <strong>SMILES:</strong>

                      </div>
                      <code className="smiles-code">
                        {moleculeSmiles}
                      </code>
                    </div>
                  )}
                </div>

                <div className="visualization-content">
                  {/* Structure 2D */}
                  <div className="graph-section">
                    <div className="graph-header">
                      <h3 className="graph-title">Structure 2D</h3>
                      <div className="graph-actions">
                        {moleculeImage && (
                          <button
                          onClick={() => {
                            const link = document.createElement('a');
                            link.href = moleculeImage;
                            link.download = `molecule_${moleculeId}_2d.png`;
                            link.click();
                          }}
                          className="action-btn"
                          title="Télécharger l'image 2D"
                        >
                          {/* Remplacez l'emoji par votre image */}
                          <img
                            src="/telecharger_image.png"
                            alt="Télécharger"
                            className="download-icon"
                            width="20"
                            height="20"
                          />
                        </button>
                        )}
                      </div>
                    </div>
                    <div className="graph-container graph-2d">
                      {moleculeImage ? (
                        <div className="molecule-image-container">
                          <img
                            src={moleculeImage}
                            alt="Structure moléculaire 2D"
                            className="molecule-2d-image"
                          />
                        </div>
                      ) : (
                        <div className="graph-placeholder">
                          <div className="loading-spinner"></div>
                          <p>Chargement de l'image 2D...</p>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Structure 3D */}
                  <div className="graph-section">
                    <div className="graph-header">
                      <h3 className="graph-title">Structure 3D Interactive</h3>
                      <div className="graph-header-right">
                        <div className="graph-actions">
                          {/* Bouton Capture 3D - Même icône que téléchargement */}
                        <button
                          onClick={exportAsPNG}
                          className="action-btn"
                          title="Télécharger la vue 3D"
                        >
                          <img
                            src="/telecharger_image.png"
                            alt="Télécharger 3D"
                            className="download-icon"
                            width="20"
                            height="20"
                          />
                        </button>
                        </div>
                      </div>
                    </div>
                    <div className="graph-container graph-3d">
                      <div
                        ref={containerRef3D}
                        className="threejs-viewer"
                        style={{
                          width: '100%',
                          height: '400px',
                          backgroundColor: '#0a0a1a',
                          borderRadius: '8px',
                          position: 'relative'
                        }}
                      />

                      {is3DLoading && (
                        <div className="loading-overlay">
                          <div className="loading-spinner"></div>
                          <p>Génération de la structure 3D...</p>
                        </div>
                      )}

                      {moleculeAnalysis && moleculeAnalysis.atoms_3d && !is3DLoading && (
                        <div className="controls-panel">
                          <div className="controls-buttons">
                            <button onClick={handleZoomIn} className="control-btn" title="Zoom +">
                              <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                              </svg>
                              Zoom +
                            </button>
                            <button onClick={handleZoomOut} className="control-btn" title="Zoom -">
                              <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18 12H6" />
                              </svg>
                              Zoom -
                            </button>
                            <button onClick={handleResetView} className="control-btn" title="Réinitialiser la vue">
                              <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                              </svg>
                              Reset
                            </button>
                            <button onClick={handleRotateLeft} className="control-btn" title="Rotation gauche">
                              ↶
                            </button>
                            <button onClick={handleRotateRight} className="control-btn" title="Rotation droite">
                              ↷
                            </button>
                            <button onClick={handleStopRotation} className="control-btn" title="Arrêter la rotation">
                              ⏹️
                            </button>
                          </div>

                          <div className="molecule-stats">
                            <div className="stat-item">
                              <span className="stat-label">Atomes:</span>
                              <span className="stat-value">{moleculeAnalysis.num_atoms_3d || 0}</span>
                            </div>
                            <div className="stat-item">
                              <span className="stat-label">Liaisons:</span>
                              <span className="stat-value">{moleculeAnalysis.bonds_3d?.length || 0}</span>
                            </div>
                            <div className="stat-item">
                              <span className="stat-label">Atomes lourds:</span>
                              <span className="stat-value">{moleculeAnalysis.properties?.heavy_atoms || 0}</span>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Propriétés prédites */}
                  {predictedProperties.length > 0 && (
                    <div className="predicted-properties">
                      <div className="properties-header">
                        <svg className="properties-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <h3 className="properties-title">Propriétés Prédites</h3>
                      </div>
                      <div className="properties-grid">
                        {predictedProperties.map((prop, index) => (
                          <div key={index} className="property-card">
                            <div className="property-value">
                              {prop}
                            </div>
                            <div className="property-target">
                              Cible: {properties[Object.keys(properties)[index]]?.toFixed(2)}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="empty-state">
                <div className="empty-state-content">
                  <div className="empty-icon">
                    <svg className="w-24 h-24" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                    </svg>
                  </div>
                  <h3 className="empty-title">Prêt à générer votre première molécule</h3>
                  <p className="empty-description">
                    Ajustez les propriétés et cliquez sur "Générer la Molécule".
                  </p>

                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Contenu caché pour le rapport PDF */}
      <ReportContent />
    </div>
  );
}

export default App;